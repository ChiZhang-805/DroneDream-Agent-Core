"""Versioned structured contracts for the complete flight-agent workflow."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from dronedream_plugin_sdk.protocol import decode_json

from .hashing import sha256_json
from .model_harness.graph import HarnessStageReceipt, HarnessTopology
from .plugin_contracts import PluginHookReceipt, PluginSnapshot

ModelRole = Literal[
    "intent_parser",
    "intent_critic",
    "plugin_router",
    "task_decomposer",
    "global_planner",
    "local_navigation_advisor",
    "plan_critic",
    "execution_monitor",
    "runtime_message_classifier",
    "completion_verifier",
    "context_summarizer",
]
SafetyAction = Literal["continue", "hold", "return", "land", "abort"]
DomainActionId = Annotated[
    str,
    Field(pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$", min_length=3, max_length=120),
]
TaskAction = DomainActionId


class StrictModel(BaseModel):
    # 这里约束字段集合及有限数值，不承诺嵌套对象不可变。跨执行边界仍须冻结并严格重验；
    # model_copy(update=...) 和列表原地修改不会自动执行完整验证。
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class Vector3(StrictModel):
    x: float
    y: float
    z: float


class QuaternionWxyz(StrictModel):
    """Unit quaternion using an explicit scalar-first convention."""

    w: float
    x: float
    y: float
    z: float

    # 功能：
    #   校验标量在前的单位四元数，不把未归一化输入自动修正为合法姿态。
    # 输入：
    #   self：待验证的 w/x/y/z 姿态分量。
    # 输出：
    #   self：归一化误差在容许范围内的姿态对象。
    @model_validator(mode="after")
    def validate_unit_quaternion(self) -> QuaternionWxyz:
        # hypot 避免有限的极端输入在平方时溢出，异常统一留在合同验证层。
        norm = math.hypot(self.w, self.x, self.y, self.z)
        if not 0.999 <= norm <= 1.001:
            raise ValueError("quaternion must be normalized")
        return self


class CalibratedRangeSensorMount(StrictModel):
    """Explicit sensor-to-body calibration; consumers must freeze their own snapshot."""

    sensor_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    translation_body_m: Vector3
    orientation_body_from_sensor: QuaternionWxyz
    minimum_range_m: float = Field(ge=0.0, le=100.0)
    maximum_range_m: float = Field(gt=0.0, le=1_000.0)

    # 功能：
    #   拒绝倒置或零宽的传感器有效测距区间。
    # 输入：
    #   self：包含安装标定及最小、最大量程的传感器合同。
    # 输出：
    #   self：量程顺序合法的标定对象。
    @model_validator(mode="after")
    def validate_range_contract(self) -> CalibratedRangeSensorMount:
        if self.maximum_range_m <= self.minimum_range_m:
            raise ValueError("maximum sensor range must exceed minimum range")
        return self


class RawRangeSample(StrictModel):
    """One range sample expressed in a calibrated sensor coordinate frame."""

    direction_sensor: Vector3
    range_m: float = Field(ge=0.0, le=1_000.0)
    hit: bool
    confidence: float = Field(ge=0.0, le=1.0)


class RawMetricRangeScan(StrictModel):
    """Pose-synchronized LiDAR/depth scan before conversion to world ENU rays."""

    sensor_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    sequence: int = Field(ge=1)
    observed_at_unix_ms: int = Field(ge=0)
    observed_at_monotonic_seconds: float = Field(ge=0.0)
    body_position_world_enu_m: Vector3
    body_orientation_world_from_body: QuaternionWxyz
    body_velocity_world_enu_mps: Vector3
    localization_covariance_m2: float = Field(ge=0.0, le=10_000.0)
    source_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    samples: list[RawRangeSample] = Field(min_length=1, max_length=250_000)
    dynamic_obstacles: list[DynamicObstacleObservation] = Field(
        default_factory=list, max_length=512
    )


class RangeRayObservation(StrictModel):
    """One calibrated metric free-space ray ending at a hit or sensor range."""

    origin_m: Vector3
    endpoint_m: Vector3
    hit: bool
    confidence: float = Field(ge=0.0, le=1.0)
    observed_at_monotonic_seconds: float = Field(ge=0.0)


class PerceptionFusionHealth(StrictModel):
    schema_version: Literal["dronedream.perception-fusion-health.v1"] = (
        "dronedream.perception-fusion-health.v1"
    )
    sensor_id: str
    latest_sequence: int = Field(ge=0)
    stream_age_seconds: float = Field(ge=0.0)
    localization_covariance_m2: float = Field(ge=0.0)
    accepted_ray_count: int = Field(ge=0)
    map_observation_count: int = Field(ge=0)
    stream_healthy: bool
    issue_codes: list[str] = Field(default_factory=list, max_length=32)


class BodyFrameControlIntent(StrictModel):
    """Short-lived, joystick-like motion request from one local control expert.

    The local expert does not author a world coordinate or a motor command.  It
    requests body-relative translational velocity and yaw rate inside an
    independently enforced vehicle/safety envelope.  The explicit model call,
    navigation snapshot and expiry bindings prevent a command from surviving a
    goal change or being replayed against fresher sensor state.
    """

    schema_version: Literal["dronedream.body-frame-control-intent.v1"] = (
        "dronedream.body-frame-control-intent.v1"
    )
    source_expert: Literal[
        "local-navigation-policy",
        "precision-maneuver-policy",
        "recovery-policy",
    ]
    control_origin: Literal[
        "candidate-path-derived",
        "continuous-model-output",
    ] = "candidate-path-derived"
    model_call_id: str = Field(pattern=r"^model-[0-9a-f]{24}$")
    # The model input and the task reference are different objects. Neither
    # may be substituted for the other merely because both are SHA-256 values.
    navigation_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at_unix_ms: int = Field(ge=0)
    valid_until_unix_ms: int = Field(ge=0)
    forward_velocity_mps: float = Field(ge=-20.0, le=20.0)
    right_velocity_mps: float = Field(ge=-20.0, le=20.0)
    up_velocity_mps: float = Field(ge=-20.0, le=20.0)
    yaw_rate_dps: float = Field(ge=-180.0, le=180.0)
    yaw_control_mode: Literal["model-rate", "route-heading-assist"] = "model-rate"
    harness_control_scale: float = Field(default=1.0, ge=0.1, le=1.0)
    maximum_acceleration_mps2: float = Field(gt=0.0, le=30.0)
    maximum_jerk_mps3: float = Field(gt=0.0, le=100.0)

    # 功能：
    #   1. 限制机体系控制的短时寿命和三轴合速度，避免逐轴合法但合速度超限。
    #   2. 禁止航向辅助与模型转速同时争用偏航控制权；实际机型及碰撞约束由下游重验。
    # 输入：
    #   self：带模型、任务、观测绑定的机体系速度及偏航率请求。
    # 输出：
    #   self：通过公共物理边界检查的控制意图。
    @model_validator(mode="after")
    def validate_control_horizon(self) -> BodyFrameControlIntent:
        if self.yaw_control_mode == "route-heading-assist" and self.yaw_rate_dps != 0.0:
            raise ValueError("heading assistance cannot also request a model yaw rate")
        horizon_ms = self.valid_until_unix_ms - self.generated_at_unix_ms
        if not 20 <= horizon_ms <= 2_000:
            raise ValueError("body-frame control horizon must be in [20, 2000] ms")
        speed_mps = math.sqrt(
            self.forward_velocity_mps**2 + self.right_velocity_mps**2 + self.up_velocity_mps**2
        )
        if speed_mps > 20.0 + 1e-9:
            raise ValueError("body-frame control velocity exceeds the physical envelope")
        return self


class NormalizedPilotControl(StrictModel):
    """Controller-neutral, continuous output of a learned local pilot.

    The four axes are deliberately dimensionless.  A vehicle-bound deterministic
    adapter maps them to body velocity and yaw-rate limits before the existing
    acceleration, jerk, collision and command-lease gates run.  This keeps a
    learned model independent of PX4/ArduPilot transport details and prevents it
    from writing motor outputs or inventing world-frame coordinates.
    """

    schema_version: Literal["dronedream.normalized-pilot-control.v1"] = (
        "dronedream.normalized-pilot-control.v1"
    )
    forward_axis: float = Field(ge=-1.0, le=1.0)
    right_axis: float = Field(ge=-1.0, le=1.0)
    up_axis: float = Field(ge=-1.0, le=1.0)
    yaw_axis: float = Field(ge=-1.0, le=1.0)


class TextNavigationDecision(StrictModel):
    schema_version: Literal["dronedream.text-navigation-decision.v1"] = (
        "dronedream.text-navigation-decision.v1"
    )
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    action: Literal[
        "select-candidate",
        "pilot-control",
        "hold",
        "request-new-scan",
        "abort",
    ]
    selected_candidate_id: str | None = None
    rationale_summary: str = Field(min_length=1, max_length=600)
    risk_notes: list[str] = Field(default_factory=list, max_length=16)
    controller_step_scale: float = Field(default=1.0, ge=0.1, le=1.0)
    pilot_control: NormalizedPilotControl | None = None

    # 功能：
    #   核对动作类型与候选身份、连续四轴输出是否配套，非运动决策不得携带运动载荷。
    # 输入：
    #   self：本地模型或云模型返回的结构化导航决策。
    # 输出：
    #   self：字段含义一致的导航决策；是否可执行仍由安全层判断。
    @model_validator(mode="after")
    def validate_candidate_binding(self) -> TextNavigationDecision:
        if self.action == "select-candidate" and not self.selected_candidate_id:
            raise ValueError("candidate selection requires selected_candidate_id")
        if self.action != "select-candidate" and self.selected_candidate_id is not None:
            raise ValueError("non-selection action cannot carry selected_candidate_id")
        if self.action == "pilot-control" and self.pilot_control is None:
            raise ValueError("pilot-control action requires continuous control axes")
        if self.action not in {"select-candidate", "pilot-control"} and (
            self.pilot_control is not None
        ):
            raise ValueError("non-motion action cannot carry pilot control")
        return self


class IndoorNavigationCycleReceipt(StrictModel):
    """Evidence for one model-advised, deterministically validated local plan cycle."""

    schema_version: Literal["dronedream.indoor-navigation-cycle.v1"] = (
        "dronedream.indoor-navigation-cycle.v1"
    )
    sequence: int = Field(ge=1)
    trigger: Literal[
        "initial",
        "periodic",
        "goal-amended",
        "progress-stalled",
        "dynamic-obstacle",
        "new-map-evidence",
    ]
    snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    perception_health: PerceptionFusionHealth
    model_action: Literal[
        "select-candidate",
        "pilot-control",
        "hold",
        "request-new-scan",
        "abort",
        "not-invoked",
    ]
    model_call_id: str | None = None
    selected_candidate_id: str | None = None
    selected_metric_path_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    controller_target_m: Vector3 | None = None
    controller_step_scale: float | None = Field(default=None, ge=0.1, le=1.0)
    deterministic_metric_path_revalidated: bool = False
    dynamic_path_revalidated: bool = False
    path_continuity_retained: bool = False
    control_authority: Literal["deterministic-local-safety"] = "deterministic-local-safety"
    hold_reason: str | None = Field(default=None, max_length=160)
    failure_stage: (
        Literal[
            "snapshot-compilation",
            "model-invocation",
            "worker-result",
        ]
        | None
    ) = None
    failure_exception_type: str | None = Field(
        default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_.]{0,159}$"
    )
    failure_root_exception_type: str | None = Field(
        default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_.]{0,159}$"
    )
    failure_reason_code: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,159}$")
    failure_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_diagnostic_metrics: dict[
        str,
        Annotated[float, Field(ge=0.0, le=60_000.0)],
    ] = Field(default_factory=dict, max_length=16)

    # 功能：
    #   只允许短 ASCII 指标名与有限数值进入失败诊断，避免日志夹带任意文本或无效耗时。
    # 输入：
    #   cls：当前回执类型。
    #   value：失败阶段的诊断指标字典。
    # 输出：
    #   value：名称及数值均合法的诊断指标。
    @field_validator("failure_diagnostic_metrics")
    @classmethod
    def validate_failure_diagnostic_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        if any(
            not key
            or len(key) > 80
            or not key.isascii()
            or not key[0].isalpha()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in key)
            or not math.isfinite(metric)
            for key, metric in value.items()
        ):
            raise ValueError("failure diagnostic metrics are invalid")
        return value


class MetricVoxelMapSnapshot(StrictModel):
    schema_version: Literal["dronedream.metric-voxel-map.v1"] = "dronedream.metric-voxel-map.v1"
    resolution_m: float = Field(gt=0.02, le=5.0)
    minimum_bound_m: Vector3
    maximum_bound_m: Vector3
    occupied_voxels: list[tuple[int, int, int]] = Field(default_factory=list, max_length=250_000)
    free_voxels: list[tuple[int, int, int]] = Field(default_factory=list, max_length=250_000)
    observation_count: int = Field(ge=0)
    latest_observation_monotonic_seconds: float | None = Field(default=None, ge=0.0)


class DynamicObstacleObservation(StrictModel):
    obstacle_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    position_m: Vector3
    velocity_mps: Vector3
    radius_m: float = Field(gt=0.0, le=20.0)
    height_m: float = Field(gt=0.0, le=50.0)
    confidence: float = Field(ge=0.0, le=1.0)
    # Retained lost detections must keep their real age, even after a long
    # safe hold. Freshness/authority limits belong to consumers, not a storage
    # cap that crashes fusion at 30 seconds or encourages rejuvenation/deletion.
    age_seconds: float = Field(ge=0.0)


class OnboardPerceptionFrame(StrictModel):
    """Calibrated world-frame perception accepted by runtime fusion."""

    schema_version: Literal["dronedream.onboard-perception-frame.v1"] = (
        "dronedream.onboard-perception-frame.v1"
    )
    sensor_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    sequence: int = Field(ge=1)
    observed_at_unix_ms: int = Field(ge=0)
    coordinate_frame: Literal["world-enu"] = "world-enu"
    localization_position_m: Vector3
    localization_velocity_mps: Vector3
    localization_covariance_m2: float = Field(ge=0.0, le=10_000.0)
    # Valid raw pixels/returns, not spatial free-space coverage. Unknown is not 100%.
    source_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    range_rays: list[RangeRayObservation] = Field(min_length=1, max_length=250_000)
    dynamic_obstacles: list[DynamicObstacleObservation] = Field(
        default_factory=list, max_length=512
    )


class LocalPlannerRequest(StrictModel):
    schema_version: Literal["dronedream.local-planner-request.v1"] = (
        "dronedream.local-planner-request.v1"
    )
    current_position_m: Vector3
    current_velocity_mps: Vector3
    current_acceleration_mps2: Vector3 | None = None
    target_position_m: Vector3
    dynamic_obstacles: list[DynamicObstacleObservation] = Field(
        default_factory=list, max_length=512
    )
    vehicle_radius_m: float = Field(gt=0.0, le=5.0)
    vehicle_height_m: float = Field(gt=0.0, le=10.0)
    max_speed_mps: float = Field(gt=0.0, le=20.0)
    max_acceleration_mps2: float = Field(gt=0.0, le=30.0)
    max_jerk_mps3: float = Field(default=100.0, gt=0.0, le=100.0)
    requested_velocity_mps: Vector3 | None = None
    requested_yaw_rate_dps: float = Field(default=0.0, ge=-180.0, le=180.0)
    required_clearance_m: float = Field(ge=0.0, le=20.0)
    prediction_horizon_seconds: float = Field(gt=0.1, le=15.0)
    prediction_step_seconds: float = Field(gt=0.02, le=1.0)
    perception_stream_healthy: bool = True
    # 年龄是事实，不截断到可接受范围；超龄观测应被安全策略拒绝，而非阻断制动请求。
    perception_stream_age_seconds: float = Field(default=0.0, ge=0.0)
    maximum_perception_age_seconds: float = Field(default=0.5, gt=0.02, le=5.0)
    localization_covariance_m2: float = Field(default=0.0, ge=0.0, le=10_000.0)
    maximum_localization_covariance_m2: float = Field(default=0.25, gt=0.0, le=100.0)


class PredictiveSafetyDecision(StrictModel):
    schema_version: Literal["dronedream.predictive-safety-decision.v1"] = (
        "dronedream.predictive-safety-decision.v1"
    )
    action: Literal["continue", "slow", "hold", "replan"]
    selected_velocity_mps: Vector3
    # Hold sends zero velocity feed-forward to the flight controller. This
    # separate quantity is the reachable next-step velocity used by the
    # braking forecast, not an instruction and not measured stopping motion.
    braking_prediction_velocity_mps: Vector3 | None = None
    selected_yaw_rate_dps: float = Field(default=0.0, ge=-180.0, le=180.0)
    maximum_acceleration_mps2: float = Field(default=1.0, gt=0.0, le=30.0)
    maximum_jerk_mps3: float = Field(default=100.0, gt=0.0, le=100.0)
    control_source: Literal[
        "route-target",
        "local-model-body-control",
        "deterministic-safety-override",
        "deterministic-brake",
    ] = "route-target"
    predicted_path_m: list[Vector3] = Field(min_length=1, max_length=1_000)
    minimum_predicted_clearance_m: float
    time_to_minimum_clearance_seconds: float = Field(ge=0.0, le=15.0)
    threat_obstacle_id: str | None = None
    evaluated_candidate_count: int = Field(ge=1, le=1_000)
    issue_codes: list[str] = Field(default_factory=list, max_length=32)

    # 功能：
    #   仅在没有制动预测时省略新增的可选字段，保留历史回执的原有摘要身份。
    # 输入：
    #   self：当前安全决策。
    #   handler：Pydantic 提供的其余字段序列化器。
    # 输出：
    #   payload：供传输及摘要计算的决策字典。
    @model_serializer(mode="wrap")
    def omit_absent_braking_forecast(self, handler):
        payload = handler(self)
        if self.braking_prediction_velocity_mps is None:
            payload.pop("braking_prediction_velocity_mps", None)
        return payload

    # 功能：
    #   将制动预测与实际指令分离：hold 只能发送零速度和零偏航，预测速度不是动作输出。
    # 输入：
    #   self：候选安全决策及可选的下一步制动预测。
    # 输出：
    #   self：制动与运动语义一致的安全决策。
    @model_validator(mode="after")
    def validate_hold_control(self) -> PredictiveSafetyDecision:
        if self.action == "hold":
            velocity = self.selected_velocity_mps
            if (
                any((velocity.x, velocity.y, velocity.z))
                or self.selected_yaw_rate_dps != 0
                or self.control_source != "deterministic-brake"
            ):
                raise ValueError("hold requires zero velocity and yaw from deterministic braking")
        elif self.braking_prediction_velocity_mps is not None:
            raise ValueError("a braking hold forecast cannot label a motion command")
        return self


class RuntimeLocalSafetyObservation(StrictModel):
    schema_version: Literal["dronedream.runtime-local-safety-observation.v1"] = (
        "dronedream.runtime-local-safety-observation.v1"
    )
    sequence: int = Field(ge=1)
    observed_at_unix_ms: int = Field(ge=0)
    source: Literal["simulation-ground-truth", "onboard", "external"]
    stream_healthy: bool
    stream_age_seconds: float = Field(ge=0.0)
    localization_covariance_m2: float = Field(ge=0.0, le=10_000.0)
    current_position_m: Vector3
    current_velocity_mps: Vector3
    current_acceleration_world_enu_mps2: Vector3 | None = None
    body_orientation_world_from_body: QuaternionWxyz | None = None
    target_position_m: Vector3
    dynamic_obstacles: list[DynamicObstacleObservation] = Field(
        default_factory=list, max_length=512
    )


class ControlObservationBudget(StrictModel):
    """Why a checked proposal retained or lost its original observation lease.

    Diagnostic evidence, not an independent permission to execute a command.
    The clearance belongs to the proposal, even if the resulting command brakes.
    """

    source_observed_at_unix_ms: int = Field(ge=0, strict=True)
    control_deadline_unix_ms: int = Field(ge=0, strict=True)
    disposition: Literal["control-eligible", "context-only", "reject"]
    reason: str = Field(min_length=1, max_length=96)
    clearance_margin_m: float = Field(ge=0)
    uncertainty_margin_m: float = Field(ge=0)
    downstream_reserve_ms: int = Field(ge=0, strict=True)


class RuntimeLocalSafetyCommand(StrictModel):
    schema_version: Literal["dronedream.runtime-local-safety-command.v1"] = (
        "dronedream.runtime-local-safety-command.v1"
    )
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_sequence: int = Field(ge=1)
    generated_at_unix_ms: int = Field(ge=0)
    valid_until_unix_ms: int = Field(ge=0)
    source: Literal["simulation-ground-truth", "onboard", "external"]
    # Short-lived alignment between the controller estimator frame and the
    # metric world frame used by perception.  A positive X value means the
    # real/world vehicle is east of the estimator's reported position, so a
    # consumer subtracts this offset from a desired world target before
    # sending it to PX4.  This is a bounded drift correction, not a global
    # localization transform.
    estimator_to_world_position_offset_m: Vector3 = Field(
        default_factory=lambda: Vector3(x=0.0, y=0.0, z=0.0)
    )
    # Bind the short-lived command to the control context that produced it.
    # The observation hash already protects the sensor snapshot; these fields
    # additionally prevent a still-valid model-navigation command from being
    # consumed after the track controller enters or leaves recovery mode.
    evaluated_target_position_m: Vector3 | None = None
    # The exact rotation used for body-to-world conversion in this safety
    # evaluation, which may be newer than the model's original observation.
    # It describes the evaluated command frame, not a refreshed model input.
    evaluated_body_orientation_world_from_body: QuaternionWxyz | None = None
    # Stable semantic goal epoch for the command.  Coordinates alone are not
    # sufficient on round trips, where outbound and return waypoints can be
    # colocated while representing different control intentions.
    navigation_goal_id: str | None = Field(default=None, min_length=1, max_length=160)
    tracking_recovery_active: bool = False
    navigation_control_authority: Literal["route-fallback", "model-required"] = "route-fallback"
    model_navigation_authorized: bool = False
    model_call_id: str | None = None
    model_selected_candidate_id: str | None = None
    model_path_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_navigation_snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_authority_reason: str | None = Field(default=None, max_length=160)
    requested_control_intent: BodyFrameControlIntent | None = None
    command_position_m: Vector3
    decision: PredictiveSafetyDecision
    observation_budget: ControlObservationBudget | None = None

    # 功能：
    #   省略不存在的新增评估姿态及观测预算，不给历史命令凭空添加证据或改变其摘要。
    # 输入：
    #   self：当前安全命令。
    #   handler：Pydantic 提供的字段序列化器。
    # 输出：
    #   payload：保留原证据身份的命令字典。
    @model_serializer(mode="wrap")
    def preserve_historical_command_identity(self, handler):
        payload = handler(self)
        if self.evaluated_body_orientation_world_from_body is None:
            payload.pop("evaluated_body_orientation_world_from_body", None)
        if self.observation_budget is None:
            payload.pop("observation_budget", None)
        return payload

    # 功能：
    #   1. 核对控制租期、观测预算和模型意图的寿命，禁止复用已失效的运动授权。
    #   2. 限制短时定位偏移，连续模型控制禁止借仿真真值偏移改写飞行轨迹。
    #   3. 验证模型调用、语义目标及输入摘要属于同一命令，不以字段存在代替安全验证。
    # 输入：
    #   self：包含安全决策、坐标对齐和模型授权证据的命令。
    # 输出：
    #   self：绑定关系与公共控制范围合法的短时命令。
    @model_validator(mode="after")
    def validate_estimator_alignment(self) -> RuntimeLocalSafetyCommand:
        horizon_ms = self.valid_until_unix_ms - self.generated_at_unix_ms
        if not 0 <= horizon_ms <= 2_000:
            raise ValueError("safety command horizon must be in [0, 2000] ms")
        # 零寿命可保留为到期证据，但执行器仍须要求当前时刻严格早于期限。
        if self.observation_budget is not None and self.decision.action != "hold":
            budget = self.observation_budget
            if (
                budget.disposition != "control-eligible"
                or budget.source_observed_at_unix_ms > self.generated_at_unix_ms
                or self.valid_until_unix_ms > budget.control_deadline_unix_ms
            ):
                raise ValueError("motion command exceeds its observation budget")
        offset = self.estimator_to_world_position_offset_m
        if math.dist((0.0, 0.0, 0.0), (offset.x, offset.y, offset.z)) > 1.0:
            raise ValueError("estimator/world alignment correction exceeds 1 metre")
        if self.navigation_control_authority == "model-required":
            if self.model_navigation_authorized and (
                self.model_call_id is None
                or self.model_path_sha256 is None
                or self.evaluated_target_position_m is None
                or self.navigation_goal_id is None
            ):
                raise ValueError("model-authorized control requires a fully bound model lease")
            if (
                self.model_navigation_authorized
                and self.model_selected_candidate_id is None
                and self.requested_control_intent is None
            ):
                raise ValueError(
                    "model-authorized control requires a candidate or direct control intent"
                )
        elif self.model_navigation_authorized:
            raise ValueError("route-fallback control cannot claim model authorization")
        if self.requested_control_intent is not None:
            if self.requested_control_intent.control_origin == "continuous-model-output" and any(
                (offset.x, offset.y, offset.z)
            ):
                raise ValueError("continuous native control forbids runtime truth-fitted offsets")
            if not self.model_navigation_authorized:
                raise ValueError("body-frame control requires model navigation authorization")
            if self.model_call_id != self.requested_control_intent.model_call_id:
                raise ValueError("body-frame control model call differs from its command lease")
            if self.model_path_sha256 != self.requested_control_intent.task_reference_sha256:
                raise ValueError("body-frame control path differs from its command lease")
            if (
                self.model_navigation_snapshot_sha256
                != self.requested_control_intent.navigation_snapshot_sha256
            ):
                raise ValueError("body-frame control input snapshot differs from its command lease")
            if not (
                self.requested_control_intent.generated_at_unix_ms
                <= self.generated_at_unix_ms
                <= self.requested_control_intent.valid_until_unix_ms
            ):
                raise ValueError("body-frame control intent expired before command publication")
            if self.valid_until_unix_ms > self.requested_control_intent.valid_until_unix_ms:
                raise ValueError("safety command cannot outlive its model control intent")
        return self


class MapRuntimeBindings(StrictModel):
    """Normalized simulator bindings required before a map can execute missions."""

    schema_version: Literal["dronedream.map-runtime-bindings.v1"] = (
        "dronedream.map-runtime-bindings.v1"
    )
    simulator: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    coordinate_frame: Literal["ENU"] = "ENU"
    vehicle_spawn: Vector3
    mission_launch_waypoint: Vector3


class VehicleAsset(StrictModel):
    schema_version: Literal["dronedream.vehicle.v1"] = "dronedream.vehicle.v1"
    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    name: str
    coordinate_frame: Literal["base_link_frd"] = "base_link_frd"
    dry_mass_kg: float = Field(gt=0.1, le=50.0)
    max_takeoff_mass_kg: float = Field(gt=0.1, le=70.0)
    body_radius_m: float = Field(ge=0.05, le=3.0)
    body_height_m: float = Field(ge=0.05, le=3.0)
    collision_center_offset_model_m: Vector3 | None = None
    max_speed_mps: float = Field(gt=0.0, le=20.0)
    max_acceleration_mps2: float = Field(gt=0.0, le=30.0)
    qualified_range_m: float = Field(gt=0.0, le=1_000_000.0)
    reserve_battery_percent: float = Field(ge=10.0, le=90.0)
    max_pickup_payload_kg: float = Field(ge=0.0, le=20.0)
    sensors: list[str] = Field(min_length=1, max_length=24)

    # 功能：
    #   校验空机重量与碰撞中心位置，缺失偏移保持未知，不替代运行资格检查。
    # 输入：
    #   self：当前机型物理参数及可选碰撞中心偏移。
    # 输出：
    #   self：质量关系与几何包络一致的机型资产。
    @model_validator(mode="after")
    def validate_collision_center(self) -> VehicleAsset:
        if self.dry_mass_kg > self.max_takeoff_mass_kg:
            raise ValueError("vehicle dry mass exceeds maximum takeoff mass")
        # 挂载机构额定载荷和起飞总重是独立限制；实际任务取二者较小值，不能改写资产额定值。
        offset = self.collision_center_offset_model_m
        if offset is None:
            return self
        if abs(offset.x) > self.body_radius_m or abs(offset.y) > self.body_radius_m:
            raise ValueError("vehicle collision center exceeds the horizontal body envelope")
        if not (0.0 < offset.z <= self.body_height_m):
            raise ValueError("vehicle collision center exceeds the vertical body envelope")
        return self


class MapNode(StrictModel):
    node_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    label: str = Field(min_length=1, max_length=160)
    position_m: Vector3
    semantic: Literal[
        "launch",
        "office",
        "stairs",
        "door",
        "corridor",
        "outdoor",
        "pickup",
    ]


class MapEdge(StrictModel):
    edge_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    from_node: str
    to_node: str
    distance_m: float = Field(gt=0.0, le=10_000.0)
    minimum_clearance_m: float = Field(ge=0.0, le=100.0)
    speed_limit_mps: float = Field(gt=0.0, le=20.0)
    bidirectional: bool = True
    qualification: Literal["flight-verified", "geometry-derived"] = "geometry-derived"
    evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class MapAsset(StrictModel):
    schema_version: Literal["dronedream.map-graph.v1"] = "dronedream.map-graph.v1"
    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    name: str
    coordinate_frame: Literal["map_enu"] = "map_enu"
    nodes: list[MapNode] = Field(min_length=2, max_length=10_000)
    edges: list[MapEdge] = Field(min_length=1, max_length=50_000)
    named_entities: dict[str, str] = Field(min_length=1, max_length=1_000)

    # 功能：
    #   验证地图节点、边身份唯一且引用可解析；不把连通图的结构合法当作飞行资格。
    # 输入：
    #   self：地图节点、边及实体到节点的对应关系。
    # 输出：
    #   self：标识及引用一致的地图图结构。
    @model_validator(mode="after")
    def validate_graph(self) -> MapAsset:
        identifiers = [node.node_id for node in self.nodes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("map node identifiers must be unique")
        known = set(identifiers)
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("map edge identifiers must be unique")
        if any(edge.from_node not in known or edge.to_node not in known for edge in self.edges):
            raise ValueError("map edge references an unknown node")
        if any(node_id not in known for node_id in self.named_entities.values()):
            raise ValueError("named entity references an unknown node")
        return self


class CatalogEntity(StrictModel):
    entity_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    aliases: list[str] = Field(min_length=1, max_length=24)
    position_m: Vector3
    semantic: str = Field(min_length=1, max_length=96)
    source_pointer: str = Field(min_length=1, max_length=240)


class MapCatalog(StrictModel):
    schema_version: Literal["dronedream.map-catalog.v1"] = "dronedream.map-catalog.v1"
    scene_id: str
    coordinate_frame: Literal["ENU"] = "ENU"
    semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entities: list[CatalogEntity] = Field(min_length=1, max_length=2_000)
    road_segment_ids: list[str] = Field(default_factory=list, max_length=2_000)
    topology_available: bool
    known_limits: list[str] = Field(default_factory=list, max_length=64)


class AttachmentArtifact(StrictModel):
    schema_version: Literal["dronedream.attachment-artifact.v1"] = (
        "dronedream.attachment-artifact.v1"
    )
    attachment_id: str = Field(pattern=r"^attachment-[0-9a-f]{32}$")
    display_name: str = Field(min_length=1, max_length=240)
    content_type: str = Field(min_length=1, max_length=160)
    size_bytes: int = Field(ge=0, le=512 * 1024 * 1024)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decoder_plugin_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    decoded_kind: Literal[
        "text",
        "document",
        "image",
        "video",
        "audio",
        "rosbag",
        "point-cloud",
        "geospatial",
        "bim",
        "cad",
        "binary-metadata",
    ]
    text: str | None = Field(default=None, max_length=200_000)
    structured_data: dict[str, Any] = Field(default_factory=dict)
    model_input: dict[str, Any] = Field(default_factory=dict)
    issue_codes: list[str] = Field(default_factory=list, max_length=32)


class MissionRequest(StrictModel):
    schema_version: Literal["dronedream.mission-request.v1"] = "dronedream.mission-request.v1"
    conversation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", max_length=96)
    message: str = Field(min_length=3, max_length=4_000)
    start_entity: str = Field(default="__auto__", min_length=1, max_length=160)
    locale: Literal["zh-CN", "en-US"] = "zh-CN"
    attachments: list[AttachmentArtifact] = Field(default_factory=list, max_length=32)
    input_channel: Literal["text", "voice", "camera", "api", "webhook", "scheduled"] = "text"
    input_metadata: dict[str, Any] = Field(default_factory=dict)

    # 功能：
    #   拒绝用户文本中的低位控制字符，保留自然语言换行和制表符。
    # 输入：
    #   cls：任务请求类型。
    #   value：去除首尾空白后的任务文本。
    # 输出：
    #   value：字符范围合法的任务文本。
    @field_validator("message")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(character) < 32 and character not in {"\n", "\t"} for character in value):
            raise ValueError("message contains control characters")
        return value


class MissionAssetPairQualificationBinding(StrictModel):
    """Portable proof that planning used one immutable, qualified asset pair."""

    schema_version: Literal["dronedream.mission-asset-pair-binding.v1"] = (
        "dronedream.mission-asset-pair-binding.v1"
    )
    qualification_id: str = Field(pattern=r"^asset-qualification-[0-9a-f]{24}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    map_asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    map_source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    map_qualified_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vehicle_asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    vehicle_source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vehicle_qualified_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_versions: dict[str, str] = Field(min_length=1, max_length=64)
    plugin_snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    qualified_at: datetime

    # 功能：
    #   检查资格环境的组件名称与版本非空，并按名称排序以稳定序列化顺序。
    # 输入：
    #   cls：资产对资格绑定类型。
    #   value：组件名称到实际版本的字典。
    # 输出：
    #   versions：排序后的环境版本字典。
    @field_validator("environment_versions")
    @classmethod
    def validate_environment_versions(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not version.strip() for key, version in value.items()):
            raise ValueError("qualification environment versions must be non-empty")
        versions = dict(sorted(value.items()))
        return versions


class IntentArtifact(StrictModel):
    schema_version: Literal["dronedream.intent.v1"] = "dronedream.intent.v1"
    goal: str = Field(min_length=3, max_length=1_000)
    start_entity: str = Field(min_length=1, max_length=160)
    target_entity: str = Field(min_length=1, max_length=160)
    return_entity: str = Field(min_length=1, max_length=160)
    payload_action: DomainActionId
    constraints: list[str] = Field(default_factory=list, max_length=32)
    missing_critical_fields: list[str] = Field(default_factory=list, max_length=16)
    assumptions: list[str] = Field(default_factory=list, max_length=16)
    environment_mode: Literal[
        "qualified-static-map",
        "known-map-with-dynamic-obstacles",
        "unknown-indoor-environment",
    ] = "qualified-static-map"


class NavigationReadinessReport(StrictModel):
    """Evidence-backed limits of the currently selected map/vehicle pair."""

    schema_version: Literal["dronedream.navigation-readiness.v1"] = (
        "dronedream.navigation-readiness.v1"
    )
    static_map_planning_ready: bool
    static_collision_geometry_ready: bool
    occupancy_esdf_ready: bool
    indoor_localization_ready: bool
    onboard_obstacle_perception_ready: bool
    dynamic_obstacle_tracking_ready: bool
    qualified_static_simulation_ready: bool
    known_dynamic_map_autonomy_ready: bool
    arbitrary_indoor_autonomy_ready: bool
    sensor_evidence: list[str] = Field(default_factory=list, max_length=32)
    issue_codes: list[str] = Field(default_factory=list, max_length=32)


class IntentCritique(StrictModel):
    schema_version: Literal["dronedream.intent-critique.v1"] = "dronedream.intent-critique.v1"
    accepted: bool
    issue_codes: list[str] = Field(default_factory=list, max_length=16)
    repair_instructions: list[str] = Field(default_factory=list, max_length=16)


class ActionDefinition(StrictModel):
    schema_version: Literal["dronedream.action-definition.v1"] = "dronedream.action-definition.v1"
    action_id: DomainActionId
    domain_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    label: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=600)
    movement: bool = False
    payload: bool = False
    flight_boundary: Literal["none", "takeoff", "landing"] = "none"
    input_schema: dict[str, Any] = Field(default_factory=dict)
    required_success_evidence: list[str] = Field(min_length=1, max_length=24)
    allowed_fallbacks: list[SafetyAction] = Field(min_length=1, max_length=6)
    simulator_executor: str = Field(min_length=3, max_length=160)
    runtime_executor: str | None = Field(default=None, max_length=160)
    authority: Literal["plan", "simulate", "control", "actuate"] = "plan"


class DomainActionCatalog(StrictModel):
    schema_version: Literal["dronedream.domain-action-catalog.v1"] = (
        "dronedream.domain-action-catalog.v1"
    )
    catalog_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    domain_ids: list[str] = Field(min_length=1, max_length=32)
    actions: list[ActionDefinition] = Field(min_length=1, max_length=256)

    # 功能：
    #   保持动作身份唯一且起飞、降落边界存在，防止插件覆盖基础任务边界。
    # 输入：
    #   self：已安装领域动作集合。
    # 输出：
    #   self：保留基础边界的动作目录。
    @model_validator(mode="after")
    def validate_actions(self) -> DomainActionCatalog:
        identifiers = [item.action_id for item in self.actions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("domain action identifiers must be unique")
        if "takeoff" not in identifiers or "land" not in identifiers:
            raise ValueError("domain catalog must retain takeoff and land boundaries")
        return self


class MissionContract(StrictModel):
    schema_version: Literal["dronedream.mission-contract.v1"] = "dronedream.mission-contract.v1"
    contract_id: str = Field(pattern=r"^mission-[0-9a-f]{24}$")
    conversation_id: str
    goal: str
    start_node: str
    target_node: str
    return_node: str
    payload_action: DomainActionId
    domain_ids: list[str] = Field(default_factory=lambda: ["core.flight"], max_length=32)
    authorized_actions: list[DomainActionId] = Field(
        default_factory=lambda: [
            "takeoff",
            "traverse",
            "navigate",
            "pickup",
            "return",
            "land",
        ],
        min_length=2,
        max_length=256,
    )
    action_catalog_sha256: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")
    map_asset_id: str
    map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    map_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vehicle_asset_id: str
    vehicle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    constraints: list[str] = Field(max_length=32)
    immutable_safety_rules: list[str] = Field(min_length=1, max_length=32)


class TaskNode(StrictModel):
    task_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=64)
    action: TaskAction
    target_node: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, max_length=16)
    success_evidence: list[str] = Field(min_length=1, max_length=16)
    max_retries: int = Field(default=1, ge=0, le=8)
    fallback: SafetyAction


class TaskGraph(StrictModel):
    schema_version: Literal["dronedream.task-graph.v1", "dronedream.task-graph.v2"] = (
        "dronedream.task-graph.v2"
    )
    revision: int = Field(default=1, ge=1)
    nodes: list[TaskNode] = Field(min_length=1, max_length=128)

    # 功能：
    #   以逐层剥离根节点的方式验证有界任务图，拒绝重复身份、未知依赖、自依赖和环。
    # 输入：
    #   self：最多 128 个节点的任务依赖图。
    # 输出：
    #   self：结构上可拓扑执行的任务图。
    @model_validator(mode="after")
    def validate_dag(self) -> TaskGraph:
        identifiers = [node.task_id for node in self.nodes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("task identifiers must be unique")
        known = set(identifiers)
        remaining: dict[str, set[str]] = {}
        for node in self.nodes:
            if node.task_id in node.depends_on:
                raise ValueError("task cannot depend on itself")
            if any(dependency not in known for dependency in node.depends_on):
                raise ValueError("task dependency is unknown")
            remaining[node.task_id] = set(node.depends_on)
        while remaining:
            roots = [node_id for node_id, dependencies in remaining.items() if not dependencies]
            if not roots:
                raise ValueError("task graph must be acyclic")
            for node_id in roots:
                del remaining[node_id]
            for dependencies in remaining.values():
                dependencies.difference_update(roots)
        return self


class TaskGraphArtifact(StrictModel):
    schema_version: Literal["dronedream.task-graph-artifact.v1"] = (
        "dronedream.task-graph-artifact.v1"
    )
    graph: TaskGraph


class SemanticPlan(StrictModel):
    schema_version: Literal["dronedream.semantic-plan.v1"] = "dronedream.semantic-plan.v1"
    ordered_targets: list[str] = Field(min_length=1, max_length=64)
    route_policy: Literal[
        "balanced",
        "clearance-first",
        "minimum-time",
        "energy-conserving",
        "few-transitions",
    ] = "balanced"
    rationale_summary: str = Field(min_length=1, max_length=600)

    # 功能：
    #   只截短不承载控制权的解释文字，避免因过长说明重试已结构正确的计划。
    # 输入：
    #   cls：语义规划类型。
    #   value：模型返回的解释字段，错误类型留给后续字段验证拒绝。
    # 输出：
    #   rationale：最多六百字符的说明或仍待字段验证的原值。
    @field_validator("rationale_summary", mode="before")
    @classmethod
    def bound_non_authoritative_rationale(cls, value: object) -> object:
        rationale = value[:600] if isinstance(value, str) else value
        return rationale


class PlannerContribution(StrictModel):
    schema_version: Literal["dronedream.planner-contribution.v1"] = (
        "dronedream.planner-contribution.v1"
    )
    planner_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    layer: Literal[
        "semantic",
        "temporal",
        "global",
        "local",
        "indoor",
        "outdoor",
        "dynamic-obstacle",
        "energy",
        "link",
        "payload",
        "regulatory",
    ]
    applicable: bool
    hard_constraints: list[str] = Field(default_factory=list, max_length=32)
    objective_metrics: list[str] = Field(default_factory=list, max_length=32)
    required_inputs: list[str] = Field(default_factory=list, max_length=32)
    deterministic_gates: dict[str, bool] = Field(min_length=1, max_length=32)


class PlannerValidation(StrictModel):
    schema_version: Literal["dronedream.planner-validation.v1"] = "dronedream.planner-validation.v1"
    planner_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    accepted: bool
    deterministic_gates: dict[str, bool] = Field(min_length=1, max_length=32)
    issue_codes: list[str] = Field(default_factory=list, max_length=32)


class RoutePoint(StrictModel):
    node_id: str
    position_m: Vector3


class PlanSegment(StrictModel):
    segment_id: str = Field(pattern=r"^segment-[0-9]{3}$")
    task_id: str
    from_node: str
    to_node: str
    path: list[RoutePoint] = Field(min_length=2, max_length=1_000)
    speed_limit_mps: float = Field(gt=0.0, le=20.0)
    minimum_clearance_m: float = Field(gt=0.0, le=100.0)
    success_evidence: list[str] = Field(min_length=1, max_length=16)


class FlightPlan(StrictModel):
    schema_version: Literal["dronedream.flight-plan.v1"] = "dronedream.flight-plan.v1"
    revision: int = Field(ge=1)
    contract_id: str
    segments: list[PlanSegment] = Field(min_length=1, max_length=256)
    semantic_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    # 功能：
    #   核对航段身份、段内起终节点及跨段几何连续性，避免同名节点掩盖位置跳变。
    # 输入：
    #   self：按执行顺序排列的完整飞行计划。
    # 输出：
    #   self：身份与几何连接一致的计划。
    @model_validator(mode="after")
    def validate_continuity(self) -> FlightPlan:
        if len({segment.segment_id for segment in self.segments}) != len(self.segments):
            raise ValueError("flight plan segment identifiers must be unique")
        for segment in self.segments:
            if (
                segment.path[0].node_id != segment.from_node
                or segment.path[-1].node_id != segment.to_node
            ):
                raise ValueError("flight plan segment path endpoints differ from its nodes")
        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if previous.to_node != current.from_node:
                raise ValueError("flight plan segments are discontinuous")
            prior = previous.path[-1].position_m
            following = current.path[0].position_m
            if (
                math.dist((prior.x, prior.y, prior.z), (following.x, following.y, following.z))
                > 1e-6
            ):
                raise ValueError(
                    "FLIGHT_PLAN_PATH_DISCONTINUITY: flight plan segment geometry is discontinuous"
                )
        return self


class PlanCritique(StrictModel):
    schema_version: Literal["dronedream.plan-critique.v1"] = "dronedream.plan-critique.v1"
    accepted: bool
    issue_codes: list[str] = Field(default_factory=list, max_length=32)
    repair_instructions: list[str] = Field(default_factory=list, max_length=32)


class RouteQuery(StrictModel):
    schema_version: Literal["dronedream.route-query.v1"] = "dronedream.route-query.v1"
    start_node: str
    goal_node: str
    require_flight_verified_edges: bool = False


class GraphRoute(StrictModel):
    schema_version: Literal["dronedream.graph-route.v1"] = "dronedream.graph-route.v1"
    start_node: str
    goal_node: str
    node_ids: list[str] = Field(min_length=1, max_length=10_000)
    edge_ids: list[str] = Field(max_length=10_000)
    positions_m: list[Vector3] = Field(min_length=1, max_length=10_000)
    route_length_m: float = Field(ge=0.0)
    all_edges_flight_verified: bool

    # 功能：
    #   验证节点、坐标和边序列逐项对齐，保留合法往返的重复节点，不自行认定飞行资格。
    # 输入：
    #   self：寻路器或插件返回的图路线。
    # 输出：
    #   self：起终点身份及序列维度一致的路线。
    @model_validator(mode="after")
    def validate_route_binding(self) -> GraphRoute:
        if (
            len(self.positions_m) != len(self.node_ids)
            or len(self.edge_ids) != len(self.node_ids) - 1
        ):
            raise ValueError("route nodes, positions and edges must be aligned")
        if self.node_ids[0] != self.start_node or self.node_ids[-1] != self.goal_node:
            raise ValueError("route endpoint identity differs from its node sequence")
        return self


class RouteCollision(StrictModel):
    sample_index: int = Field(ge=0)
    position_m: Vector3
    primitive_name: str
    clearance_m: float


class RouteClearanceReport(StrictModel):
    schema_version: Literal["dronedream.route-clearance.v1"] = "dronedream.route-clearance.v1"
    accepted: bool
    route_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_interval_m: float = Field(gt=0.0)
    sample_count: int = Field(ge=1)
    primitive_count: int = Field(ge=1)
    collision_count: int = Field(ge=0)
    minimum_clearance_m: float
    minimum_clearance_point: Vector3
    minimum_clearance_primitive: str
    segment_minimum_clearances_m: list[float] = Field(default_factory=list, max_length=9_999)
    collisions: list[RouteCollision] = Field(default_factory=list, max_length=100)


class RouteAlternativeCandidate(StrictModel):
    schema_version: Literal["dronedream.route-alternative-candidate.v1"] = (
        "dronedream.route-alternative-candidate.v1"
    )
    alternative_id: str
    strategy_tool_id: str
    equivalent_strategy_tool_ids: list[str] = Field(default_factory=list, max_length=16)
    route: GraphRoute
    clearance: RouteClearanceReport
    objectives: dict[str, float]
    hard_gates: dict[str, bool]
    feasible: bool
    issue_codes: list[str] = Field(default_factory=list, max_length=32)


class RouteAlternativeSet(StrictModel):
    schema_version: Literal["dronedream.route-alternative-set.v1"] = (
        "dronedream.route-alternative-set.v1"
    )
    contract_id: str
    candidates: list[RouteAlternativeCandidate] = Field(min_length=1, max_length=16)
    objective_weights: dict[str, float]


class RouteAlternativeDecision(StrictModel):
    schema_version: Literal["dronedream.route-alternative-decision.v1"] = (
        "dronedream.route-alternative-decision.v1"
    )
    selected_alternative_id: str
    ranked_alternative_ids: list[str] = Field(min_length=1, max_length=16)
    normalized_scores: dict[str, float]
    selection_reasons: list[str] = Field(min_length=1, max_length=16)
    rejected_alternatives: dict[str, list[str]] = Field(default_factory=dict)


class Px4TrackPoint(StrictModel):
    x: float
    y: float
    z: float
    phase: Literal["launch", "transit", "stairs", "pickup", "return", "land"]
    speed_limit_mps: float = Field(gt=0.0, le=20.0)


class WorldTrackPoint(StrictModel):
    east_m: float
    north_m: float
    up_m: float


class Px4CoordinateContract(StrictModel):
    source: Literal["Gazebo ENU vehicle-collision-envelope center"] = (
        "Gazebo ENU vehicle-collision-envelope center"
    )
    executor_x: Literal["PX4 local north physically mapped to qualified-map ENU north"] = (
        "PX4 local north physically mapped to qualified-map ENU north"
    )
    executor_y: Literal["PX4 local east physically mapped to qualified-map ENU east"] = (
        "PX4 local east physically mapped to qualified-map ENU east"
    )
    executor_z: Literal["PX4 local up"] = "PX4 local up"
    model_root_world_enu_m: list[float] = Field(min_length=3, max_length=3)
    collision_center_offset_model_m: list[float] = Field(
        min_length=3,
        max_length=3,
    )

    # 功能：
    #   限制显式碰撞中心偏移，拒绝地面以下或超过通用支持范围的机体几何。
    # 输入：
    #   self：世界、机体根节点与碰撞中心的坐标转换合同。
    # 输出：
    #   self：偏移范围合法的坐标合同。
    @model_validator(mode="after")
    def validate_collision_offset(self) -> Px4CoordinateContract:
        east, north, up = self.collision_center_offset_model_m
        if abs(east) > 5.0 or abs(north) > 5.0 or not (0.0 < up <= 5.0):
            raise ValueError("collision-center offset is outside the supported envelope")
        return self

    # 功能：
    #   将已验证的碰撞中心偏移展开为固定三元组，保持原有米制单位及轴序。
    # 输入：
    #   self：当前坐标转换合同。
    # 输出：
    #   offset：模型局部坐标系 X、Y、Z 偏移三元组，单位米。
    def resolved_collision_center_offset_model_m(self) -> tuple[float, float, float]:
        east, north, up = self.collision_center_offset_model_m
        offset = (float(east), float(north), float(up))
        return offset


class Px4Track(StrictModel):
    schema_version: Literal["dronedream.px4-track.v2"] = "dronedream.px4-track.v2"
    track_type: Literal["custom"] = "custom"
    coordinate_contract: Px4CoordinateContract
    points: list[Px4TrackPoint] = Field(min_length=2, max_length=10_000)
    source_world_points: list[WorldTrackPoint] = Field(min_length=2, max_length=10_000)
    stop_at_waypoints: bool = True
    waypoint_hold_seconds: float = Field(ge=0.0, le=30.0)
    waypoint_position_tolerance_m: float = Field(default=0.2, gt=0.0, le=2.0)
    waypoint_speed_tolerance_mps: float = Field(default=0.15, gt=0.0, le=2.0)
    waypoint_stable_window_seconds: float = Field(default=0.5, gt=0.0, le=10.0)
    waypoint_settle_timeout_seconds: float = Field(default=12.0, gt=0.0, le=120.0)

    # 功能：
    #   保持执行器点与世界点数量相同；具体坐标换算与几何净空由固定内核另外核验。
    # 输入：
    #   self：导出的 PX4 轨迹及来源世界点。
    # 输出：
    #   self：点序列维度对应的轨迹。
    @model_validator(mode="after")
    def aligned_points(self) -> Px4Track:
        if len(self.points) != len(self.source_world_points):
            raise ValueError("PX4 and world point arrays must be aligned")
        return self


class Px4TrackRequest(StrictModel):
    schema_version: Literal["dronedream.px4-track-request.v1"] = "dronedream.px4-track-request.v1"
    route: GraphRoute
    waypoint_hold_seconds: float = Field(default=0.4, ge=0.0, le=30.0)


class RuntimeTrackRequest(StrictModel):
    schema_version: Literal["dronedream.runtime-track-request.v1"] = (
        "dronedream.runtime-track-request.v1"
    )
    route: GraphRoute
    prior_track: Px4Track
    target_node: str
    vehicle: VehicleAsset


class RuntimeAssessment(StrictModel):
    schema_version: Literal["dronedream.runtime-assessment.v1"] = "dronedream.runtime-assessment.v1"
    action: Literal["accept", "retry", "replan", "hold", "abort"]
    issue_codes: list[str] = Field(default_factory=list, max_length=32)
    repair_hint: str | None = Field(default=None, max_length=400)


class RuntimeCheckpoint(StrictModel):
    checkpoint_id: str = Field(pattern=r"^checkpoint-[0-9]{3,6}$")
    segment_id: str = Field(pattern=r"^segment-[0-9]{3,6}$")
    task_id: str
    track_point_index: int = Field(ge=1)
    target_node: str


class RuntimeCheckpointContract(StrictModel):
    schema_version: Literal["dronedream.runtime-checkpoints.v1"] = (
        "dronedream.runtime-checkpoints.v1"
    )
    contract_id: str
    checkpoints: list[RuntimeCheckpoint] = Field(min_length=1, max_length=128)


class RuntimeCheckpointRequest(StrictModel):
    schema_version: Literal["dronedream.runtime-checkpoint-request.v1"] = (
        "dronedream.runtime-checkpoint-request.v1"
    )
    contract_id: str
    checkpoint: RuntimeCheckpoint
    observed_position_ned_m: Vector3
    observed_velocity_ned_mps: Vector3
    commanded_position_ned_m: Vector3
    position_error_m: float = Field(ge=0.0)
    speed_mps: float = Field(ge=0.0)
    battery_percent: float = Field(ge=0.0, le=100.0)
    deterministic_gates: dict[str, bool] = Field(min_length=1, max_length=32)


class RuntimeCheckpointDecision(StrictModel):
    schema_version: Literal["dronedream.runtime-checkpoint-decision.v1"] = (
        "dronedream.runtime-checkpoint-decision.v1"
    )
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment: RuntimeAssessment
    model_call: ModelCallRecord
    continue_authorized: bool
    plugin_hook_receipts: list[PluginHookReceipt] = Field(default_factory=list)


class RuntimeActionAdapterDefinition(StrictModel):
    """Materialized plugin adapter that can execute one or more domain actions."""

    schema_version: Literal["dronedream.runtime-action-adapter.v1"] = (
        "dronedream.runtime-action-adapter.v1"
    )
    adapter_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$", max_length=160)
    runtime_executors: list[str] = Field(min_length=1, max_length=64)
    driver: Literal[
        "mavsdk-camera",
        "gazebo-payload",
        "payload-transition",
        "ros2-service",
    ]
    default_parameters: dict[str, Any] = Field(default_factory=dict)
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=15.0, gt=0.0, le=300.0)
    authority: Literal["simulate", "control", "actuate"]


class RuntimeActionAdapterCatalog(StrictModel):
    schema_version: Literal["dronedream.runtime-action-adapter-catalog.v1"] = (
        "dronedream.runtime-action-adapter-catalog.v1"
    )
    catalog_id: str = Field(pattern=r"^runtime-actions\.[0-9a-f]{24}$")
    adapters: list[RuntimeActionAdapterDefinition] = Field(min_length=1, max_length=128)

    # 功能：
    #   限定适配器身份和执行器归属，拒绝重复声明或跨适配器抢占同一执行器。
    # 输入：
    #   self：本次物化的运行时动作适配器目录。
    # 输出：
    #   self：执行器所有权无歧义的目录。
    @model_validator(mode="after")
    def validate_executor_ownership(self) -> RuntimeActionAdapterCatalog:
        identifiers = [adapter.adapter_id for adapter in self.adapters]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("runtime adapter identifiers must be unique")
        owners: dict[str, str] = {}
        for adapter in self.adapters:
            if len(set(adapter.runtime_executors)) != len(adapter.runtime_executors):
                raise ValueError("runtime adapter executors must be unique")
            for executor in adapter.runtime_executors:
                previous = owners.get(executor)
                if previous is not None and previous != adapter.adapter_id:
                    raise ValueError(f"runtime executor is owned by multiple adapters: {executor}")
                owners[executor] = adapter.adapter_id
        return self


class RuntimeActionExecutionStep(StrictModel):
    schema_version: Literal["dronedream.runtime-action-step.v1"] = (
        "dronedream.runtime-action-step.v1"
    )
    step_id: str = Field(pattern=r"^action-[0-9]{3,6}$")
    task_id: str
    action: DomainActionId
    target_node: str
    trigger: Literal["post-takeoff", "post-replan", "checkpoint"]
    checkpoint_id: str | None = Field(default=None, pattern=r"^checkpoint-[0-9]{3,6}$")
    depends_on: list[str] = Field(default_factory=list, max_length=16)
    runtime_executor: str
    adapter_id: str
    driver: Literal[
        "mavsdk-camera",
        "gazebo-payload",
        "payload-transition",
        "ros2-service",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)
    required_success_evidence: list[str] = Field(min_length=1, max_length=24)
    max_attempts: int = Field(ge=1, le=9)
    fallback: SafetyAction
    timeout_seconds: float = Field(gt=0.0, le=300.0)
    authority: Literal["simulate", "control", "actuate"]

    # 功能：
    #   保持检查点触发与起飞后、重规划后触发互斥，避免动作在错误阶段执行。
    # 输入：
    #   self：单个运行时动作及触发绑定。
    # 输出：
    #   self：触发方式与检查点字段一致的动作步骤。
    @model_validator(mode="after")
    def validate_trigger_binding(self) -> RuntimeActionExecutionStep:
        if self.trigger == "checkpoint" and self.checkpoint_id is None:
            raise ValueError("checkpoint-triggered action requires checkpoint_id")
        if self.trigger in {"post-takeoff", "post-replan"} and self.checkpoint_id is not None:
            raise ValueError(f"{self.trigger} action cannot bind a checkpoint_id")
        return self


class RuntimeActionExecutionContract(StrictModel):
    schema_version: Literal["dronedream.runtime-action-execution.v1"] = (
        "dronedream.runtime-action-execution.v1"
    )
    contract_id: str
    task_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain_action_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    steps: list[RuntimeActionExecutionStep] = Field(default_factory=list, max_length=128)


class MissionVerificationRequirement(StrictModel):
    """One evidence requirement compiled from the accepted task and control contracts."""

    schema_version: Literal["dronedream.mission-verification-requirement.v1"] = (
        "dronedream.mission-verification-requirement.v1"
    )
    requirement_id: str = Field(pattern=r"^verification-[0-9]{3,6}$")
    phase: Literal["pre-confirmation", "runtime", "completion"]
    subject_type: Literal[
        "mission",
        "task",
        "route",
        "track",
        "runtime-action",
        "landing",
    ]
    subject_id: str = Field(min_length=1, max_length=160)
    evidence_key: str = Field(min_length=1, max_length=240)
    evidence_authority: Literal[
        "deterministic-core",
        "qualified-tool",
        "runtime-checkpoint",
        "runtime-action-adapter",
        "px4-telemetry",
        "completion-verifier",
    ]
    source_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_before: Literal["user-confirmation", "task-advance", "mission-completion"]
    blocking: Literal[True] = True
    model_only_evidence_accepted: Literal[False] = False


class MissionVerificationPlan(StrictModel):
    """Hash-bound verification matrix reviewed before the user can confirm execution."""

    schema_version: Literal["dronedream.mission-verification-plan.v1"] = (
        "dronedream.mission-verification-plan.v1"
    )
    contract_id: str = Field(pattern=r"^mission-[0-9a-f]{24}$")
    plan_revision: int = Field(ge=1)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    flight_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_route_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    route_clearance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    px4_track_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_checkpoints_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_actions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deterministic_gates: dict[str, bool] = Field(min_length=1, max_length=32)
    requirements: list[MissionVerificationRequirement] = Field(min_length=1, max_length=1_024)

    # 功能：
    #   要求验证矩阵覆盖确认前、运行中和完成后三个阶段，拒绝失败门控及重复证据要求。
    # 输入：
    #   self：由当前合同和轨迹编译的验证矩阵。
    # 输出：
    #   self：结构完整且门控通过的验证计划。
    @model_validator(mode="after")
    def validate_verification_matrix(self) -> MissionVerificationPlan:
        identifiers = [item.requirement_id for item in self.requirements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("verification requirement identifiers must be unique")
        if not all(self.deterministic_gates.values()):
            raise ValueError("mission verification plan contains a failed deterministic gate")
        phases = {item.phase for item in self.requirements}
        if phases != {"pre-confirmation", "runtime", "completion"}:
            raise ValueError("mission verification plan must cover every lifecycle phase")
        identities = {
            (item.phase, item.subject_type, item.subject_id, item.evidence_key)
            for item in self.requirements
        }
        if len(identities) != len(self.requirements):
            raise ValueError("mission verification plan contains duplicate evidence requirements")
        return self


class RuntimeActionExecutionReceipt(StrictModel):
    schema_version: Literal["dronedream.runtime-action-receipt.v1"] = (
        "dronedream.runtime-action-receipt.v1"
    )
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step_id: str = Field(pattern=r"^action-[0-9]{3,6}$")
    task_id: str
    action: DomainActionId
    adapter_id: str
    runtime_executor: str
    status: Literal["accepted", "rejected"]
    attempts: int = Field(ge=1, le=9)
    started_at: datetime
    completed_at: datetime
    output: dict[str, Any] = Field(default_factory=dict)
    observed_success_evidence: list[str] = Field(default_factory=list, max_length=24)
    deterministic_gates: dict[str, bool] = Field(min_length=1, max_length=32)
    issue_codes: list[str] = Field(default_factory=list, max_length=16)


class TaskThread(StrictModel):
    """Stable identity and lifecycle for one user-visible mission conversation."""

    schema_version: Literal["dronedream.task-thread.v1"] = "dronedream.task-thread.v1"
    conversation_id: str
    mission_id: str = Field(pattern=r"^mission-[0-9a-f]{32}$")
    state: Literal[
        "planning",
        "awaiting_confirmation",
        "executing",
        "holding",
        "landing",
        "completed",
        "failed",
    ]
    current_plan_revision_id: str | None = Field(default=None, pattern=r"^plan-[0-9a-f]{32}$")
    active_execution_id: str | None = Field(default=None, pattern=r"^execution-[0-9a-f]{32}$")
    created_at: datetime
    updated_at: datetime


class PlanRevisionRecord(StrictModel):
    """One replaceable plan inside a stable task thread."""

    schema_version: Literal["dronedream.plan-revision.v1"] = "dronedream.plan-revision.v1"
    plan_revision_id: str = Field(pattern=r"^plan-[0-9a-f]{32}$")
    conversation_id: str
    mission_id: str = Field(pattern=r"^mission-[0-9a-f]{32}$")
    revision: int = Field(ge=1)
    parent_plan_revision_id: str | None = Field(default=None, pattern=r"^plan-[0-9a-f]{32}$")
    status: Literal["proposed", "superseded", "confirmed", "executing", "completed", "failed"]
    contract_id: str
    prepared_mission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_message_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime


class MissionLifecycleBinding(StrictModel):
    """Sidecar that avoids changing an already hash-bound PreparedMission schema."""

    schema_version: Literal["dronedream.mission-lifecycle-binding.v1"] = (
        "dronedream.mission-lifecycle-binding.v1"
    )
    thread: TaskThread
    plan_revision: PlanRevisionRecord


class RuntimeControlSession(StrictModel):
    schema_version: Literal["dronedream.runtime-control-session.v1"] = (
        "dronedream.runtime-control-session.v1"
    )
    conversation_id: str
    mission_id: str = Field(pattern=r"^mission-[0-9a-f]{32}$")
    plan_revision_id: str = Field(pattern=r"^plan-[0-9a-f]{32}$")
    contract_id: str
    execution_id: str = Field(pattern=r"^execution-[0-9a-f]{32}$")
    prepared_mission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: Literal["accepting", "closed"] = "accepting"
    created_at: datetime


class RuntimeUserMessage(StrictModel):
    schema_version: Literal["dronedream.runtime-user-message.v1"] = (
        "dronedream.runtime-user-message.v1"
    )
    message_id: str = Field(pattern=r"^runtime-msg-[0-9a-f]{32}$")
    conversation_id: str
    mission_id: str = Field(pattern=r"^mission-[0-9a-f]{32}$")
    plan_revision_id: str = Field(pattern=r"^plan-[0-9a-f]{32}$")
    contract_id: str
    execution_id: str = Field(pattern=r"^execution-[0-9a-f]{32}$")
    text: str = Field(min_length=1, max_length=4_000)
    submitted_at: datetime

    # 功能：
    #   拒绝实时用户消息中的低位控制字符，保持可读换行及制表符。
    # 输入：
    #   cls：实时用户消息类型。
    #   value：本次用户修订文本。
    # 输出：
    #   value：字符范围合法的消息文本。
    @field_validator("text")
    @classmethod
    def reject_message_control_characters(cls, value: str) -> str:
        if any(ord(character) < 32 and character not in {"\n", "\t"} for character in value):
            raise ValueError("runtime message contains control characters")
        return value


class RuntimeTrackProgress(StrictModel):
    """Executor-owned next waypoint; geometric proximity cannot establish route progress."""

    track_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    next_track_point_index: int = Field(strict=True, ge=1, le=9_999)


class RuntimeHoldAcknowledgement(StrictModel):
    schema_version: Literal["dronedream.runtime-hold-ack.v1"] = "dronedream.runtime-hold-ack.v1"
    message_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    message_id: str = Field(pattern=r"^runtime-msg-[0-9a-f]{32}$")
    execution_id: str = Field(pattern=r"^execution-[0-9a-f]{32}$")
    interrupted_phase: Literal[
        "TAKEOFF", "TRACK", "WAYPOINT_SETTLE", "CHECKPOINT", "ACTION", "PAUSED"
    ]
    schedule_index: int | None = Field(default=None, ge=0)
    # 老回执可以作为历史资料读取，但缺少进度绑定时不能授权截取剩余路线。
    track_progress: RuntimeTrackProgress | None = None
    detected_at: datetime
    detection_latency_ms: int = Field(ge=0)
    stable_at: datetime
    stabilization_latency_ms: int = Field(ge=0)
    frozen_command_ned_m: Vector3
    hold_command_ned_m: Vector3
    observed_position_ned_m: Vector3
    observed_velocity_ned_mps: Vector3
    position_error_m: float = Field(ge=0.0)
    speed_mps: float = Field(ge=0.0)
    side_effects_inhibited: Literal[True] = True
    deterministic_gates: dict[str, bool] = Field(min_length=1, max_length=32)


class RuntimeMessageClassification(StrictModel):
    schema_version: Literal["dronedream.runtime-message-classification.v1"] = (
        "dronedream.runtime-message-classification.v1"
    )
    message_kind: Literal[
        "emergency_stop",
        "mission_amendment",
        "motion_adjustment",
        "informational",
    ]
    requested_action: Literal[
        "land",
        "replan",
        "adjust_motion",
        "resume",
        "redirect",
        "set_speed",
        "pause",
        "return_home",
        "safe_land",
        "set_return_point",
        "set_coverage",
        "camera_control",
        "payload_control",
        "set_avoidance",
        "follow_target",
        "operator_takeover",
        "operator_release",
    ]
    target_entity: str | None = Field(default=None, max_length=160)
    requires_plan_revision: bool
    summary: str = Field(min_length=1, max_length=400)
    parameters: dict[str, Any] = Field(default_factory=dict)


class RuntimeAmendmentDirective(StrictModel):
    schema_version: Literal["dronedream.runtime-amendment-directive.v1"] = (
        "dronedream.runtime-amendment-directive.v1"
    )
    action: str = Field(pattern=r"^[a-z][a-z0-9._-]{1,63}$")
    parameters: dict[str, Any] = Field(default_factory=dict)
    requires_stable_hold: bool = True
    requires_plan_revision: bool = True
    requires_core_authorization: bool = True
    issue_codes: list[str] = Field(default_factory=list, max_length=32)


class RuntimeInterruptionDecision(StrictModel):
    schema_version: Literal["dronedream.runtime-interruption-decision.v1"] = (
        "dronedream.runtime-interruption-decision.v1"
    )
    message_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hold_ack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    classification: RuntimeMessageClassification
    model_call: ModelCallRecord
    authorized_action: Literal[
        "resume_original", "hold", "hold_for_replan", "apply_command", "land"
    ]
    authorization_gates: dict[str, bool] = Field(min_length=1, max_length=32)
    decision_reason: str = Field(min_length=1, max_length=400)
    plugin_hook_receipts: list[PluginHookReceipt] = Field(default_factory=list)
    amendment_directive: RuntimeAmendmentDirective | None = None


class RuntimeToolReceipt(StrictModel):
    tool_id: str
    plugin_id: str | None = None
    plugin_package_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    outcome: Literal["accepted", "rejected", "failed"]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CoveragePlanRequest(StrictModel):
    schema_version: Literal["dronedream.coverage-plan-request.v1"] = (
        "dronedream.coverage-plan-request.v1"
    )
    center_enu_m: Vector3
    polygon_enu_m: list[Vector3] = Field(default_factory=list, max_length=128)
    width_m: float = Field(default=6.0, ge=1.0, le=5000.0)
    height_m: float = Field(default=6.0, ge=1.0, le=5000.0)
    lane_spacing_m: float = Field(ge=0.2, le=100.0)
    boundary_margin_m: float = Field(default=0.5, ge=0.0, le=50.0)
    altitude_m: float = Field(ge=0.2, le=500.0)

    # 功能：
    #   当调用方指定覆盖多边形时要求至少三个顶点，空列表保留矩形覆盖模式。
    # 输入：
    #   self：矩形或多边形覆盖规划参数。
    # 输出：
    #   self：顶点数量符合所选模式的规划请求。
    @model_validator(mode="after")
    def validate_polygon(self) -> CoveragePlanRequest:
        if self.polygon_enu_m and len(self.polygon_enu_m) < 3:
            raise ValueError("coverage polygon requires at least three vertices")
        return self


class CoveragePattern(StrictModel):
    schema_version: Literal["dronedream.coverage-pattern.v1"] = "dronedream.coverage-pattern.v1"
    pattern: Literal["lawnmower", "inward-spiral"]
    points_enu_m: list[Vector3] = Field(min_length=2, max_length=10_000)
    lane_count: int = Field(ge=1, le=10_000)
    estimated_area_m2: float = Field(gt=0.0)
    deterministic_gates: dict[str, bool] = Field(min_length=1, max_length=32)


class RuntimeAuthorizedCommand(StrictModel):
    """A hash-bound peripheral or flight-policy command issued only from stable hold."""

    schema_version: Literal["dronedream.runtime-authorized-command.v1"] = (
        "dronedream.runtime-authorized-command.v1"
    )
    message_id: str = Field(pattern=r"^runtime-msg-[0-9a-f]{32}$")
    execution_id: str = Field(pattern=r"^execution-[0-9a-f]{32}$")
    action: Literal["camera_control", "payload_control", "set_avoidance"]
    parameters: dict[str, Any]
    message_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hold_ack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deterministic_gates: dict[str, bool] = Field(min_length=1, max_length=32)
    plugin_hook_receipts: list[PluginHookReceipt] = Field(default_factory=list)
    generated_at: datetime


class RuntimeCommandAdoption(StrictModel):
    schema_version: Literal["dronedream.runtime-command-adoption.v1"] = (
        "dronedream.runtime-command-adoption.v1"
    )
    artifact_kind: Literal["runtime-command"] = "runtime-command"
    message_id: str = Field(pattern=r"^runtime-msg-[0-9a-f]{32}$")
    execution_id: str = Field(pattern=r"^execution-[0-9a-f]{32}$")
    action: Literal["camera_control", "payload_control", "set_avoidance"]
    command_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    success: Literal[True] = True
    observed_result: dict[str, Any]
    adopted_at: datetime


class RuntimeOperatorTakeoverGrant(StrictModel):
    """One-time, hash-bound authority for bounded manual velocity commands."""

    schema_version: Literal["dronedream.runtime-operator-takeover-grant.v1"] = (
        "dronedream.runtime-operator-takeover-grant.v1"
    )
    message_id: str = Field(pattern=r"^runtime-msg-[0-9a-f]{32}$")
    execution_id: str = Field(pattern=r"^execution-[0-9a-f]{32}$")
    operator_id: str = Field(min_length=1, max_length=160)
    message_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hold_ack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grant_token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_horizontal_speed_mps: float = Field(gt=0.0, le=3.0)
    maximum_vertical_speed_mps: float = Field(gt=0.0, le=2.0)
    maximum_yaw_rate_dps: float = Field(gt=0.0, le=180.0)
    deterministic_gates: dict[str, bool] = Field(min_length=1, max_length=32)
    issued_at: datetime
    expires_at: datetime

    # 功能：
    #   要求接管授权使用有时区的绝对时间，且过期时间严格晚于签发时间。
    # 输入：
    #   self：人工接管授权及其签发、过期时间。
    # 输出：
    #   self：时序合法的接管授权。
    @model_validator(mode="after")
    def expiry_after_issue(self) -> RuntimeOperatorTakeoverGrant:
        if self.expires_at.utcoffset() is None or self.issued_at.utcoffset() is None:
            raise ValueError("operator takeover times require a timezone")
        if self.expires_at <= self.issued_at:
            raise ValueError("operator takeover grant must expire after it is issued")
        return self


class RuntimeOperatorControlCommand(StrictModel):
    """Short-lived manual command written only after the app authenticates a grant token."""

    schema_version: Literal["dronedream.runtime-operator-control-command.v1"] = (
        "dronedream.runtime-operator-control-command.v1"
    )
    message_id: str = Field(pattern=r"^runtime-msg-[0-9a-f]{32}$")
    execution_id: str = Field(pattern=r"^execution-[0-9a-f]{32}$")
    grant_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence: int = Field(ge=1)
    action: Literal["velocity", "release"]
    velocity_ned_mps: Vector3 = Field(default_factory=lambda: Vector3(x=0.0, y=0.0, z=0.0))
    yaw_rate_dps: float = Field(ge=-180.0, le=180.0)
    duration_seconds: float = Field(gt=0.0, le=0.5)
    issued_at: datetime


class RuntimeOperatorTakeoverAdoption(StrictModel):
    schema_version: Literal["dronedream.runtime-operator-takeover-adoption.v1"] = (
        "dronedream.runtime-operator-takeover-adoption.v1"
    )
    message_id: str = Field(pattern=r"^runtime-msg-[0-9a-f]{32}$")
    execution_id: str = Field(pattern=r"^execution-[0-9a-f]{32}$")
    grant_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    success: Literal[True] = True
    adopted_at: datetime


class RuntimeReplacementTrack(StrictModel):
    """A code-validated track that may supersede the active track during stable hold."""

    schema_version: Literal[
        "dronedream.runtime-replacement-track.v1",
        "dronedream.runtime-replacement-track.v2",
    ] = "dronedream.runtime-replacement-track.v2"
    message_id: str = Field(pattern=r"^runtime-msg-[0-9a-f]{32}$")
    execution_id: str = Field(pattern=r"^execution-[0-9a-f]{32}$")
    replacement_sequence: int = Field(ge=1)
    message_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hold_ack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_track_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    amendment_action: Literal[
        "replan",
        "redirect",
        "set_speed",
        "return_home",
        "set_return_point",
        "set_coverage",
        "follow_target",
    ] = "replan"
    amendment_parameters: dict[str, Any] = Field(default_factory=dict)
    target_node: str
    return_node: str
    route: GraphRoute
    clearance: RouteClearanceReport
    track: Px4Track
    revised_task_graph: TaskGraph | None = None
    runtime_checkpoints: RuntimeCheckpointContract | None = None
    runtime_actions: RuntimeActionExecutionContract | None = None
    superseded_runtime_action_step_ids: list[str] = Field(
        default_factory=list,
        max_length=128,
    )
    plugin_tool_receipts: list[RuntimeToolReceipt] = Field(default_factory=list)
    plugin_hook_receipts: list[PluginHookReceipt] = Field(default_factory=list)
    deterministic_gates: dict[str, bool] = Field(min_length=1, max_length=32)
    generated_at: datetime

    # 功能：
    #   1. 运行中若替换任务图，要求检查点及动作合同成套更新，不能新旧混用。
    #   2. 核对合同、任务图摘要、动作检查点及被替代步骤的身份关系。
    # 输入：
    #   self：稳定悬停时生成的候选替换轨迹和执行修订。
    # 输出：
    #   self：执行修订完整且引用一致的替换包。
    @model_validator(mode="after")
    def validate_execution_revision(self) -> RuntimeReplacementTrack:
        revision_parts = (
            self.revised_task_graph,
            self.runtime_checkpoints,
            self.runtime_actions,
        )
        if any(part is not None for part in revision_parts) and not all(
            part is not None for part in revision_parts
        ):
            raise ValueError("runtime replacement execution revision must be complete")
        if self.runtime_checkpoints is None or self.runtime_actions is None:
            if self.superseded_runtime_action_step_ids:
                raise ValueError("superseded runtime actions require a complete revision")
            return self
        if (
            self.runtime_checkpoints.contract_id != self.runtime_actions.contract_id
            or self.runtime_actions.task_graph_sha256 != sha256_json(self.revised_task_graph)
        ):
            raise ValueError("runtime replacement execution revision identity mismatch")
        checkpoint_ids = {
            checkpoint.checkpoint_id for checkpoint in self.runtime_checkpoints.checkpoints
        }
        if any(
            step.checkpoint_id not in checkpoint_ids
            for step in self.runtime_actions.steps
            if step.trigger == "checkpoint"
        ):
            raise ValueError("runtime replacement action references unknown checkpoint")
        if len(self.superseded_runtime_action_step_ids) != len(
            set(self.superseded_runtime_action_step_ids)
        ):
            raise ValueError("duplicate superseded runtime action step id")
        return self


class CompletionAssessment(StrictModel):
    schema_version: Literal["dronedream.completion-assessment.v1"] = (
        "dronedream.completion-assessment.v1"
    )
    accepted: bool
    issue_codes: list[str] = Field(default_factory=list, max_length=32)


class ConversationEvent(StrictModel):
    schema_version: Literal["dronedream.conversation-event.v1"] = "dronedream.conversation-event.v1"
    event_id: str = Field(pattern=r"^event-[0-9a-f]{24}$")
    conversation_id: str
    sequence: int = Field(ge=1)
    role: Literal["user", "assistant", "tool", "system"]
    event_type: str = Field(min_length=1, max_length=96)
    payload: dict[str, Any]
    created_at: datetime


class ConversationWindow(StrictModel):
    schema_version: Literal["dronedream.conversation-window.v1"] = (
        "dronedream.conversation-window.v1"
    )
    conversation_id: str
    summary: str | None = None
    recent_events: list[ConversationEvent]
    previous_response_ids: dict[str, str] = Field(default_factory=dict)


class PluginInvocation(StrictModel):
    tool_id: str = Field(pattern=r"^[a-z][a-z0-9._-]*$")
    arguments_json: str = Field(min_length=2, max_length=65_536)
    purpose: str = Field(min_length=1, max_length=300)

    # 功能：
    #   有界解析插件参数，拒绝重复键、非有限数值、过深结构及非对象顶层。
    # 输入：
    #   self：包含模型选择工具及原始 JSON 参数文本的调用计划。
    # 输出：
    #   value：通过标准 JSON 与字节预算检查的独立参数字典。
    def parsed_arguments(self) -> dict[str, Any]:
        value = decode_json(self.arguments_json, limit=65_536)
        if not isinstance(value, dict):
            raise ValueError("plugin invocation arguments_json must encode an object")
        return value


class PluginInvocationPlan(StrictModel):
    schema_version: Literal["dronedream.plugin-invocation-plan.v1"] = (
        "dronedream.plugin-invocation-plan.v1"
    )
    calls: list[PluginInvocation] = Field(default_factory=list, max_length=16)

    # 功能：
    #   限定单次路由阶段每个工具最多调用一次，禁止重复工具绕过阶段调用预算。
    # 输入：
    #   self：模型给出的有界工具调用清单。
    # 输出：
    #   self：无重复工具身份的调用计划。
    @model_validator(mode="after")
    def unique_tools(self) -> PluginInvocationPlan:
        tool_ids = [call.tool_id for call in self.calls]
        if len(tool_ids) != len(set(tool_ids)):
            raise ValueError("a plugin tool may be invoked only once per routing stage")
        return self


class ToolReceipt(StrictModel):
    schema_version: Literal["dronedream.tool-receipt.v1"] = "dronedream.tool-receipt.v1"
    call_id: str = Field(pattern=r"^tool-[0-9a-f]{24}$")
    tool_id: str = Field(pattern=r"^[a-z][a-z0-9._-]*$")
    tool_version: str
    plugin_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9._-]*$")
    plugin_package_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    outcome: Literal["accepted", "rejected", "failed"]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output: dict[str, Any]
    issue_codes: list[str] = Field(default_factory=list, max_length=32)


class LocalExpertInferenceTrace(StrictModel):
    """Per-expert votes behind one conservative local-policy decision."""

    schema_version: Literal["dronedream.local-expert-inference-trace.v1"] = (
        "dronedream.local-expert-inference-trace.v1"
    )
    requested_navigation_role: Literal[
        "local-navigation-policy",
        "precision-maneuver-policy",
        "recovery-policy",
    ]
    selected_navigation_role: Literal[
        "local-navigation-policy",
        "precision-maneuver-policy",
        "recovery-policy",
    ]
    fallback_used: bool
    navigation_action: Literal[
        "select-candidate",
        "pilot-control",
        "hold",
        "request-new-scan",
        "abort",
    ]
    navigation_selected_candidate_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
    )
    candidate_scores: list[float] = Field(min_length=8, max_length=8)
    action_scores: list[float] = Field(min_length=3, max_length=4)
    navigation_risk_score: float = Field(ge=0.0, le=1.0)
    advisory_risk_scores: dict[
        Literal[
            "risk-critic",
            "perception-health-critic",
            "settle-stability-critic",
            "payload-dynamics-adapter",
            "state-anomaly-detector",
            "cross-modal-consistency-critic",
        ],
        Annotated[float, Field(ge=0.0, le=1.0)],
    ] = Field(default_factory=dict, max_length=6)
    expert_latency_ms: dict[str, Annotated[float, Field(ge=0.0, le=60_000.0)]] = Field(
        default_factory=dict,
        max_length=8,
    )
    pipeline_latency_ms: dict[
        str,
        Annotated[float, Field(ge=0.0, le=60_000.0)],
    ] = Field(default_factory=dict, max_length=8)
    aggregate_risk_score: float = Field(ge=0.0, le=1.0)
    controller_step_scale: float = Field(ge=0.1, le=1.0)
    pilot_control: NormalizedPilotControl | None = None

    # 功能：
    #   1. 核对连续控制、候选选择和非运动动作分别携带正确的输出字段。
    #   2. 限定耗时指标的专家及流水线名称，并保持总风险不低于任何已记录专家风险。
    # 输入：
    #   self：本地专家的一次推理归因、分数及耗时记录。
    # 输出：
    #   self：字段配套且风险聚合保守的推理记录。
    @model_validator(mode="after")
    def validate_trace(self) -> LocalExpertInferenceTrace:
        if self.navigation_action == "pilot-control" and len(self.action_scores) != 4:
            raise ValueError("pilot-control expert trace requires the continuous action score")
        if self.navigation_action not in {"select-candidate", "pilot-control"} and (
            self.pilot_control is not None
        ):
            raise ValueError("non-motion expert trace cannot carry pilot control")
        if self.navigation_action == "pilot-control" and self.pilot_control is None:
            raise ValueError("pilot-control expert trace requires continuous axes")
        allowed_latency_roles = {
            "perception-encoder",
            "local-navigation-policy",
            "precision-maneuver-policy",
            "recovery-policy",
            "risk-critic",
            "perception-health-critic",
            "settle-stability-critic",
            "payload-dynamics-adapter",
            "state-anomaly-detector",
            "cross-modal-consistency-critic",
        }
        if not set(self.expert_latency_ms).issubset(allowed_latency_roles):
            raise ValueError("local expert trace contains an unknown latency role")
        if any(not math.isfinite(value) for value in self.expert_latency_ms.values()):
            raise ValueError("local expert trace latencies must be finite")
        allowed_pipeline_stages = {
            "tensor-assembly",
            "visual-input-preprocess",
            "visual-prefetch-wait",
            "backend-wall",
            "port-input-preparation",
            "port-decision-record",
            "port-wall",
        }
        if not set(self.pipeline_latency_ms).issubset(allowed_pipeline_stages):
            raise ValueError("local expert trace contains an unknown pipeline stage")
        if any(not math.isfinite(value) for value in self.pipeline_latency_ms.values()):
            raise ValueError("local expert pipeline latencies must be finite")
        if any(not math.isfinite(value) for value in [*self.candidate_scores, *self.action_scores]):
            raise ValueError("local expert trace scores must be finite")
        if (self.navigation_action == "select-candidate") != (
            self.navigation_selected_candidate_id is not None
        ):
            raise ValueError("local expert trace candidate identity must match navigation action")
        if self.aggregate_risk_score + 1e-7 < self.navigation_risk_score:
            raise ValueError("aggregate local risk cannot be lower than navigation risk")
        if any(
            self.aggregate_risk_score + 1e-7 < score for score in self.advisory_risk_scores.values()
        ):
            raise ValueError("aggregate local risk cannot be lower than an advisor vote")
        return self


class SimulationTrainingTrace(StrictModel):
    """A learner proposal, not a qualified package or fabricated inference score."""

    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_navigation_role: Literal[
        "local-navigation-policy",
        "precision-maneuver-policy",
        "recovery-policy",
    ]
    simulation_only: Literal[True] = True
    flight_qualification_granted: Literal[False] = False


class ModelCallRecord(StrictModel):
    schema_version: Literal["dronedream.model-call.v1"] = "dronedream.model-call.v1"
    call_id: str = Field(pattern=r"^model-[0-9a-f]{24}$")
    role: ModelRole
    attempt: int = Field(ge=1, le=100)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_schema: str
    provider: str
    model: str
    response_id: str | None = None
    previous_response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)
    created_at: datetime
    local_expert_trace: LocalExpertInferenceTrace | None = None
    simulation_training_trace: SimulationTrainingTrace | None = None

    # 功能：
    #   没有训练提议时不增加训练字段，以保持既有部署推理回执的摘要身份。
    # 输入：
    #   self：当前模型调用记录。
    #   handler：Pydantic 提供的字段序列化器。
    # 输出：
    #   payload：用于存档及摘要计算的调用字典。
    @model_serializer(mode="wrap")
    def preserve_existing_call_identity(self, handler):
        payload = handler(self)
        if self.simulation_training_trace is None:
            payload.pop("simulation_training_trace", None)
        return payload

    # 功能：
    #   将仿真学习提议限定在训练提供方和本地导航角色，禁止同时冒充已部署专家推理。
    # 输入：
    #   self：带可选训练轨迹与部署推理轨迹的调用记录。
    # 输出：
    #   self：训练与部署归因互斥的调用记录。
    @model_validator(mode="after")
    def exclusive_training_authority(self):
        if self.simulation_training_trace is not None and (
            self.local_expert_trace is not None
            or self.provider != "simulation-training"
            or self.role != "local_navigation_advisor"
        ):
            raise ValueError("SIMULATION_TRAINING_CANNOT_CLAIM_DEPLOYED_INFERENCE")
        return self


class EvidenceRecord(StrictModel):
    schema_version: Literal["dronedream.evidence-record.v1"] = "dronedream.evidence-record.v1"
    sequence: int = Field(ge=1)
    created_at: datetime
    event_type: str = Field(min_length=1, max_length=96)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: dict[str, Any]


class PreparedMission(StrictModel):
    # Older schemas remain parseable only so archived evidence can be audited.
    # Every newly constructed or actively executed mission uses v4.
    schema_version: Literal[
        "dronedream.prepared-mission.v1",
        "dronedream.prepared-mission.v2",
        "dronedream.prepared-mission.v3",
        "dronedream.prepared-mission.v4",
    ] = "dronedream.prepared-mission.v4"
    status: Literal["awaiting_confirmation"] = "awaiting_confirmation"
    intent: IntentArtifact
    intent_critique: IntentCritique
    contract: MissionContract
    domain_actions: DomainActionCatalog
    simulation_capabilities: dict[str, Any] = Field(default_factory=dict)
    task_graph: TaskGraph
    semantic_plan: SemanticPlan
    plan: FlightPlan
    plan_critique: PlanCritique
    execution_route: GraphRoute
    route_clearance: RouteClearanceReport
    px4_track: Px4Track
    runtime_checkpoints: RuntimeCheckpointContract | None = None
    runtime_actions: RuntimeActionExecutionContract | None = None
    verification_plan: MissionVerificationPlan | None = None
    planning_attempts: int = Field(ge=1)
    model_attempt_count: int = Field(ge=0)
    model_calls: list[ModelCallRecord]
    plugin_snapshot: PluginSnapshot
    harness_topology: HarnessTopology
    harness_stage_receipts: list[HarnessStageReceipt]
    plugin_hook_receipts: list[PluginHookReceipt] = Field(default_factory=list)
    tool_receipts: list[ToolReceipt]
    evidence: list[EvidenceRecord]

    # 功能：
    #   当前任务包必须包含检查点、动作及验证合同；旧包仅保留历史读取，执行另行拒绝。
    # 输入：
    #   self：候选准备完成的任务包。
    # 输出：
    #   self：当前格式要件完整或可供历史读取的任务包。
    @model_validator(mode="after")
    def require_compiled_verification_for_current_packages(self) -> PreparedMission:
        if self.schema_version != "dronedream.prepared-mission.v4":
            return self
        if self.runtime_checkpoints is None:
            raise ValueError("current prepared mission requires runtime checkpoints")
        if self.runtime_actions is None:
            raise ValueError("current prepared mission requires a runtime action contract")
        if self.verification_plan is None:
            raise ValueError("current prepared mission requires a verification plan")
        return self


class Px4UlogArtifact(StrictModel):
    path: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Px4GazeboArtifactBindings(StrictModel):
    world_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    render_world_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    static_render_batching: dict[str, Any] | None = None
    semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vehicle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    route_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    track_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clearance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    controller_params_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    executor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Failed preflight runs may terminate before PX4 opens a ULog.  The
    # ``px4_ulog_present`` gate still requires at least one log for success,
    # while the evidence contract remains parseable for diagnosis.
    px4_ulogs: list[Px4UlogArtifact]
    ros_workspace: str
    model_navigation_cycles_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_navigation_calls_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_navigation_snapshots_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_navigation_timing_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    depth_safety_history_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime_evidence_writer_summary_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    runtime_snapshot_writer_summary_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    local_safety_executor_history_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    control_applications_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    learning_observations_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    learning_observation_summary_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    control_application_verification: dict[str, Any] | None = None
    local_control_timing_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    development_fault_injection_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    local_policy_selection_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    multimodal_dataset_root: str | None = None
    multimodal_dataset_records_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    semantic_label_map_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_navigation_visual_frames: list[Px4UlogArtifact] = Field(default_factory=list)


class Px4GazeboGates(StrictModel):
    # 新门控仅在对应执行链实际启用时参与验收，历史证据缺字段不能伪造一项测量。
    native_observer_subscriptions_closed: bool = False
    native_sensor_subscriptions_closed: bool = False
    model_navigation_shutdown_complete: bool = False
    offline_training_authority_absent: bool = True
    learning_observation_recording_complete: bool = False
    teacher_execution_recording_complete: bool = False
    actual_model_control_input_deadlines_bounded: bool = False
    continuous_control_effective_timing_bounded: bool = False
    independent_native_state_publication_drained: bool = False
    local_policy_qualified_selection_recorded: bool = False
    executor_completed: bool
    offboard_timing_complete: bool
    runtime_pose_samples_present: bool
    ros_observations_present: bool
    goal_observed: bool
    landing_confirmed: bool
    native_terminal_lifecycle_published: bool
    no_live_abort: bool
    px4_ulog_present: bool
    static_route_clearance_bound: bool
    local_safety_observation_present: bool = False
    local_safety_command_present: bool = False
    local_safety_observation_sequence_advanced: bool = False
    controlled_vehicle_entity_selected: bool = False
    controlled_vehicle_identity_confirmed: bool = False
    live_depth_perception_healthy: bool = False
    live_depth_metric_map_present: bool = False
    live_depth_safety_history_present: bool = False
    runtime_evidence_writer_complete: bool = False
    runtime_snapshot_writer_complete: bool = False
    model_navigation_cycle_recorded: bool = False
    model_navigation_provider_call_recorded: bool = False
    model_navigation_primary_provider_call_recorded: bool = False
    model_navigation_provider_cadence_bounded: bool = False
    model_navigation_invocation_failure_absent: bool = False
    model_navigation_fresh_revalidation_stable: bool = False
    model_navigation_snapshot_recorded: bool = False
    model_navigation_shadow_only: bool = False
    model_navigation_authorized_control_applied: bool = False
    model_navigation_visual_frame_recorded: bool = False
    model_navigation_control_authority_required: bool = False
    model_navigation_authorized_schedule_advance_recorded: bool = False
    model_navigation_route_fallback_absent: bool = False
    local_policy_simulation_admission_recorded: bool = False
    development_fault_injection_absent: bool = True
    development_payload_collection_absent: bool = True
    multimodal_dataset_summary_present: bool = False
    multimodal_dataset_records_present: bool = False
    multimodal_dataset_summary_consistent: bool = False
    semantic_supervision_recorded_for_every_sample: bool = False


class LiveSafetyEvent(StrictModel):
    """Typed fail-closed record emitted by the live Gazebo safety monitor."""

    schema_version: Literal["dronedream.live-safety-event.v1"]
    reason: Literal[
        "LIVE_STATIC_COLLISION",
        "LIVE_ROUTE_DEVIATION",
        "CONTROLLED_ENTITY_IDENTITY_MISMATCH",
        "PX4_TRACKING_TELEMETRY_STALE",
        "GAZEBO_POSE_PROCESSING_FAILED",
        "LIVE_RUNTIME_PHASE_CLOCK_INVALID",
        "LIVE_RUNTIME_PHASE_UNAVAILABLE",
        "LIVE_RUNTIME_PHASE_READ_FAILED",
    ]
    elapsed_s: float | None = Field(default=None, ge=0.0)
    vehicle_collision_center_world_enu_m: WorldTrackPoint | None = None
    error_type: str | None = Field(default=None, min_length=1, max_length=128)
    phase_read: dict[str, Any] | None = None
    reference_polyline_world_enu_m: list[tuple[float, float, float]] | None = None
    clearance_m: float | None = None
    route_deviation_m: float | None = Field(default=None, ge=0.0)
    collision_primitive: str | None = None
    nearest_collision_primitive: str | None = None
    runtime_phase: str | None = None
    goal_departure_observed: bool | None = None
    goal_observed: bool | None = None
    px4_collision_center_world_enu_m: Vector3 | None = None
    position_disagreement_m: float | None = Field(default=None, ge=0.0)
    time_aligned_position_disagreement_m: float | None = Field(default=None, ge=0.0)
    identity_offset_innovation_m: float | None = Field(default=None, ge=0.0)
    maximum_offset_innovation_m: float | None = Field(default=None, gt=0.0)
    maximum_absolute_correction_m: float | None = Field(default=None, gt=0.0, le=1.0)
    identity_alignment_seconds: float | None = Field(default=None, ge=-0.5, le=0.5)
    selected_entity_name: str | None = None
    pose_reference: str | None = None
    # 负年龄表示来源时间在未来；保留异常事实，不能夹成零并伪装为新鲜观测。
    tracking_age_seconds: float | None = None

    # 功能：
    #   按碰撞、偏航或身份/遥测异常的具体原因要求相应证据，避免只有原因标签没有观测。
    # 输入：
    #   self：实时仿真监视器报告的安全事件。
    # 输出：
    #   self：包含对应原因所需测量字段的安全事件。
    @model_validator(mode="after")
    def validate_reason_specific_evidence(self) -> LiveSafetyEvent:
        if self.reason == "GAZEBO_POSE_PROCESSING_FAILED":
            if not self.error_type:
                raise ValueError("pose processing failure requires an error type")
            return self
        if self.reason.startswith("LIVE_RUNTIME_PHASE_"):
            if self.phase_read is None or self.phase_read.get("issue") != self.reason:
                raise ValueError("phase failure requires matching phase evidence")
            return self
        if self.elapsed_s is None or self.vehicle_collision_center_world_enu_m is None:
            raise ValueError("physical safety event requires observation time and position")
        if self.reason == "LIVE_STATIC_COLLISION":
            if self.clearance_m is None or self.route_deviation_m is None:
                raise ValueError("static-collision safety evidence requires geometry metrics")
            if not self.collision_primitive:
                raise ValueError("static-collision safety evidence requires collision_primitive")
        elif self.reason == "LIVE_ROUTE_DEVIATION":
            if self.clearance_m is None or self.route_deviation_m is None:
                raise ValueError("route-deviation safety evidence requires geometry metrics")
            if not self.nearest_collision_primitive:
                raise ValueError(
                    "route-deviation safety evidence requires nearest_collision_primitive"
                )
        elif (
            self.px4_collision_center_world_enu_m is None
            or self.position_disagreement_m is None
            or not self.selected_entity_name
            or not self.pose_reference
            or self.tracking_age_seconds is None
        ):
            raise ValueError("controlled-entity mismatch requires cross-source identity evidence")
        return self


class Px4RuntimeAbortRequest(StrictModel):
    reason: str
    world_paused: bool = False
    action: str | None = None
    contract_id: str | None = None
    issue_codes: list[str] = Field(default_factory=list)
    observation_sequence: int | None = Field(default=None, ge=0)
    observation_age_ms: int | None = Field(default=None, ge=0)
    deadline_ms: int | None = Field(default=None, ge=0)
    severity: int | None = Field(default=None, ge=0)
    source: str | None = None
    safety_event: LiveSafetyEvent | None = None


class ControlledVehicleEntityEvidence(StrictModel):
    schema_version: Literal["dronedream.controlled-vehicle-entity.v1"]
    vehicle_model_name: str
    selected_entity_name: str
    pose_reference: str
    candidate_entity_names: list[str] = Field(min_length=1)
    collision_center_offset_model_m: tuple[float, float, float]
    selected_at_elapsed_s: float = Field(ge=0.0)


class ControlledVehicleIdentityEvidence(StrictModel):
    schema_version: Literal["dronedream.controlled-vehicle-identity.v1"]
    selected_entity_name: str
    pose_reference: str
    gazebo_collision_center_world_enu_m: Vector3
    aligned_gazebo_collision_center_world_enu_m: Vector3 | None = None
    px4_collision_center_world_enu_m: Vector3
    position_disagreement_m: float = Field(ge=0.0)
    time_aligned_position_disagreement_m: float | None = Field(default=None, ge=0.0)
    identity_offset_innovation_m: float | None = Field(default=None, ge=0.0)
    maximum_offset_innovation_m: float | None = Field(default=None, gt=0.0)
    maximum_absolute_correction_m: float | None = Field(default=None, gt=0.0, le=1.0)
    # Signed because either transport can lead the other. The runtime alignment
    # algorithm is independently bounded to this half-second freshness window.
    identity_alignment_seconds: float = Field(default=0.0, ge=-0.5, le=0.5)
    maximum_correction_m: float | None = Field(default=None, gt=0.0, le=1.0)
    tracking_age_seconds: float
    tracking_fresh: bool = True
    correction_source: Literal[
        "live", "cached", "unavailable", "validated-before-terminal-phase"
    ] = "unavailable"
    last_correction_age_seconds: float | None = None
    gazebo_pose_queue_age_seconds: float | None = Field(default=None, ge=0.0)
    consecutive_mismatch_samples: int = Field(ge=0)
    identity_required: bool = True
    runtime_phase: str | None = None
    validated_during_active_flight: bool = False
    accepted: bool
    updated_at_unix_ms: int = Field(ge=0)


class Px4GazeboLocalSafetyMeasurements(StrictModel):
    supported_by_executor: bool
    observation_sequence: int = Field(ge=0)
    required_clearance_m: float = Field(gt=0.0)
    identity_correction_limit_m: float | None = Field(default=None, gt=0.0, le=1.0)
    observation_artifact: str | None = None
    command_artifact: str | None = None
    authority_source: Literal["live-depth-metric-fusion", "simulation-ground-truth"] = (
        "simulation-ground-truth"
    )
    depth_perception_health: dict[str, Any] | None = None
    depth_safety_history_count: int = Field(default=0, ge=0)
    runtime_evidence_writer_summary: dict[str, Any] | None = None
    runtime_snapshot_writer_summary: dict[str, Any] | None = None
    runtime_evidence_artifact_counts: dict[str, int] | None = None
    perception_control_completion: dict[str, Any] | None = None
    executor_history_artifact: str | None = None
    executor_history_count: int = Field(default=0, ge=0)


class Px4GazeboModelControlAuthorityEvidence(StrictModel):
    control_application_counts: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    required: bool = False
    authorized_control_applied_count: int = Field(default=0, ge=0)
    authorized_schedule_advance_count: int = Field(default=0, ge=0)
    route_fallback_schedule_advance_count: int = Field(default=0, ge=0)
    unauthorized_hold_applied_count: int = Field(default=0, ge=0)
    unauthorized_schedule_hold_count: int = Field(default=0, ge=0)
    progress_schedule_hold_count: int = Field(default=0, ge=0)


class Px4GazeboLocalPolicySelection(StrictModel):
    map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_id: str = Field(min_length=1)
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_receipt_id: str = Field(min_length=1)
    rollback_package_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    scope: Literal["general", "map-specific"]
    selection_reason: str = Field(min_length=1)
    sensor_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_only: bool
    vehicle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_payload_collection: bool = False
    flight_qualification_granted: bool | None = None

    # 功能：
    #   开发采集模式必须明确否认飞行资格，不能借模型包选择记录冒充产品验收。
    # 输入：
    #   self：当前本地策略包及采集模式选择记录。
    # 输出：
    #   self：开发采集与飞行资格声明一致的策略选择。
    @model_validator(mode="after")
    def _development_collection_never_grants_flight_qualification(
        self,
    ) -> Px4GazeboLocalPolicySelection:
        if self.development_payload_collection and self.flight_qualification_granted is not False:
            raise ValueError(
                "development payload collection must explicitly deny flight qualification"
            )
        return self


class Px4GazeboModelNavigationMeasurements(StrictModel):
    enabled: bool
    provider: str | None = None
    fallback_provider: str | None = None
    primary_provider_call_count: int = Field(default=0, ge=0)
    fallback_provider_call_count: int = Field(default=0, ge=0)
    cycle_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    snapshot_count: int = Field(default=0, ge=0)
    motion_proposal_count: int = Field(default=0, ge=0)
    authorized_control_applied_count: int = Field(ge=0)
    model_timeout_seconds: float = Field(gt=0.0)
    fallback_model_timeout_seconds: float | None = Field(default=None, gt=0.0)
    cycle_period_seconds: float = Field(gt=0.0)
    control_authority: Literal["deterministic-local-safety"]
    model_control_authority_required: bool = False
    model_control_authority_evidence: Px4GazeboModelControlAuthorityEvidence = Field(
        default_factory=Px4GazeboModelControlAuthorityEvidence
    )
    controller_step_m: float | None = Field(default=None, gt=0.0)
    visual_enabled: bool
    visual_frame_count: int = Field(ge=0)
    maximum_cycle_gap_seconds: float | None = Field(default=None, ge=0.0)
    maximum_provider_participation_gap_seconds: float | None = Field(default=None, ge=0.0)
    provider_participation_gap_limit_seconds: float = Field(default=60.0, gt=0.0)
    invocation_failure_count: int = Field(default=0, ge=0)
    invocation_timeout_count: int = Field(default=0, ge=0)
    revalidation_timeout_count: int = Field(default=0, ge=0)
    perception_changed_during_model_call_count: int = Field(default=0, ge=0)
    local_policy_selection: Px4GazeboLocalPolicySelection | None = None


class Px4GazeboMultimodalDatasetMeasurements(StrictModel):
    enabled: bool
    flight_id: str | None = None
    maximum_mib: float | None = Field(default=None, gt=0.0)
    record_count: int = Field(default=0, ge=0)
    record_period_seconds: float | None = Field(default=None, gt=0.0)
    semantic_record_count: int = Field(default=0, ge=0)
    semantic_topic: str | None = None
    summary: dict[str, Any] | None = None


class Px4GazeboDevelopmentPayloadCollectionMeasurements(StrictModel):
    enabled: bool = False
    development_only: bool = False
    flight_qualification_granted: bool | None = None
    maximum_controller_step_scale: float | None = Field(default=None, ge=0.1, le=0.2)
    fresh_px4_dynamics_required: bool = False


class Px4GazeboMeasurements(StrictModel):
    native_sensor_deployment: dict[str, Any] | None = None
    simulation_learning: dict[str, Any] | None = None
    pose_sample_count: int = Field(ge=0)
    ros_observation_rows: int = Field(ge=0)
    minimum_goal_distance_m: float | None = Field(default=None, ge=0.0)
    goal_departure_observed: bool = False
    goal_observed_runtime: bool = False
    live_safety_event: LiveSafetyEvent | None = None
    external_abort_request: Px4RuntimeAbortRequest | None = None
    landing_state: str | None = None
    abort_reason: str | None = None
    executor_return_code: int | None = None
    tolerated_landing_contact_samples: int = Field(default=0, ge=0)
    minimum_tolerated_landing_clearance_m: float | None = None
    local_safety: Px4GazeboLocalSafetyMeasurements | None = None
    model_navigation: Px4GazeboModelNavigationMeasurements | None = None
    multimodal_dataset: Px4GazeboMultimodalDatasetMeasurements | None = None
    controlled_vehicle_entity: ControlledVehicleEntityEvidence | None = None
    controlled_vehicle_identity: ControlledVehicleIdentityEvidence | None = None
    development_fault_injection: dict[str, Any] | None = None
    development_payload_collection: Px4GazeboDevelopmentPayloadCollectionMeasurements | None = None


class Px4GazeboRunEvidence(StrictModel):
    schema_version: Literal["dronedream.generic-px4-gazebo-run.v1"]
    status: Literal["verified", "failed"]
    world: str
    vehicle: str
    gates: Px4GazeboGates
    measurements: Px4GazeboMeasurements
    artifacts: Px4GazeboArtifactBindings


class SimulationWorkflowResult(StrictModel):
    schema_version: Literal["dronedream.simulation-workflow-result.v1"] = (
        "dronedream.simulation-workflow-result.v1"
    )
    status: Literal["verified", "failed"]
    contract_id: str
    prepared_mission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_evidence: Px4GazeboRunEvidence
    completion_assessment: CompletionAssessment
    completion_model_call: ModelCallRecord
    checkpoint_decisions: list[RuntimeCheckpointDecision] = Field(default_factory=list)
    runtime_action_receipts: list[RuntimeActionExecutionReceipt] = Field(default_factory=list)
    runtime_interruption_decisions: list[RuntimeInterruptionDecision] = Field(default_factory=list)
    plugin_hook_receipts: list[PluginHookReceipt] = Field(default_factory=list)
    workflow_evidence_chain_head: str = Field(pattern=r"^[0-9a-f]{64}$")
