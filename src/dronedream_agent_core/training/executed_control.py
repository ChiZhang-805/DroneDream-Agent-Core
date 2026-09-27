"""Recover continuous demonstration labels from actual transport receipts.

Geometry/planner proposals, position-control feed-forward and measured aircraft
velocity are not joystick labels. This compiler accepts only acknowledged pure
velocity commands with an explicit rate-to-heading conversion. Independent
mission verification and dataset splitting remain the caller's responsibility.
"""

import math

from dronedream_plugin_sdk.protocol import copy_json

from ..contracts import NormalizedPilotControl, QuaternionWxyz, RuntimeLocalSafetyCommand, Vector3
from ..control_execution_evidence import ControlApplicationRecord, validate_application_binding
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..hashing import sha256_json
from ..pilot_control_mapping import PilotControlLimits
from ..realtime_feature_encoders import RealtimeFeatureSnapshot, world_enu_to_body
from .flight_environment import PilotAction, SafetyPositionControl

NATIVE_ACTION_COORDINATE_FRAME = "evaluated-control-body"

# 功能：
#   重新校验并独立持有实际回执及原始命令，再检查其精确绑定，不授予任何运动权限。
# 输入：
#   command：经安全层批准的命令实例。
#   application：传输层实际接受的回执实例。
# 输出：
#   command：独立且验证后的命令。
#   application：独立且验证后的回执。
def _control_evidence(command, application):
    if not isinstance(command, RuntimeLocalSafetyCommand) or not isinstance(
        application, ControlApplicationRecord
    ):
        raise ValueError("TRAINING_CONTROL_EVIDENCE_TYPE_INVALID")
    command = RuntimeLocalSafetyCommand.model_validate(command.model_dump(mode="python"))
    application = ControlApplicationRecord.model_validate(application.model_dump(mode="python"))
    validate_application_binding(command, application)
    return command, application


# 功能：
#   重新构造物理尺度，拒绝绕过冻结 dataclass 后产生的非法值。
# 输入：
#   limits：部署时绑定的物理尺度。
# 输出：
#   limits：重新通过有限性和物理范围检查的独立尺度。
def _validated_limits(limits: PilotControlLimits) -> PilotControlLimits:
    if not isinstance(limits, PilotControlLimits):
        raise ValueError("pilot control limits must use their declared contract")
    limits = PilotControlLimits(limits.horizontal_speed_mps, limits.vertical_speed_mps,
                                limits.yaw_rate_dps)
    return limits


# 功能：
#   1. 从真实回执恢复训练动作，区分四轴速度控制、位置恢复与实际安全悬停。
#   2. 拒绝过期运动；过期命令触发的安全悬停只记为干预，不能续期模型控制。
# 输入：
#   snapshot：原始模型输入快照。
#   command：原始批准命令。
#   application：实际接受的传输回执。
#   limits：当前部署的物理控制尺度。
# 输出：
#   applied：实际四轴动作或明确区分的位置及速度前馈动作。
#   expired：是否记录了过期命令后的安全悬停。
def executed_training_action(snapshot: dict, command: RuntimeLocalSafetyCommand,
                             application: ControlApplicationRecord, *, limits: PilotControlLimits
                             ) -> tuple[PilotAction | SafetyPositionControl, bool]:
    command, application = _control_evidence(command, application)
    limits = _validated_limits(limits)
    if application.accepted_at_unix_ms < command.generated_at_unix_ms:
        raise ValueError("PX4_TRAINING_RECEIPT_PRECEDES_COMMAND")
    expired = application.accepted_at_unix_ms > command.valid_until_unix_ms
    if application.transport == "velocity-ned":
        if expired:
            raise ValueError("PX4_TRAINING_RECEIPT_OUTSIDE_COMMAND_LIFETIME")
        control = executed_pilot_control(snapshot, command, application, limits=limits,
                                          execution_frame=True)
        # 序列化包含契约身份字段，只有以下四个具名轴是动作，不能依赖字典字段顺序。
        applied = PilotAction(mode="pilot-control", axes=[
            control.forward_axis, control.right_axis, control.up_axis, control.yaw_axis,
        ])
        return applied, expired
    if application.safety_action == "replan":
        selected = command.decision.selected_velocity_mps
        if (expired or application.position_ned_m is None
                or application.control_source != "deterministic-brake"
                or command.decision.selected_yaw_rate_dps != 0
                or application.yaw_rate_application is not None
                or any(abs(actual - expected) > 1e-6 for actual, expected in zip(
                    application.velocity_ned_mps, (selected.y, selected.x, -selected.z),
                    strict=True))):
            raise ValueError("PX4_TRAINING_REPLAN_RECEIPT_NOT_VERIFIABLE")
        applied = SafetyPositionControl(
            position_ned_m=application.position_ned_m,
            velocity_feedforward_ned_mps=application.velocity_ned_mps,
            yaw_heading_deg=application.yaw_heading_deg, safety_action="replan")
        return applied, expired
    if (application.position_ned_m is None or any(application.velocity_ned_mps)
            or application.safety_action != "hold"
            or application.control_source != "deterministic-brake"
            or any(command.decision.selected_velocity_mps.model_dump().values())
            or command.decision.selected_yaw_rate_dps != 0
            or application.yaw_rate_application is not None):
        raise ValueError("PX4_TRAINING_HOLD_RECEIPT_NOT_VERIFIABLE")
    applied = PilotAction(mode="hold", axes=[0.] * 4)
    return applied, expired


# 功能：
#   1. 在固定 NED 坐标系中计算实际速度前馈与航向的归一化变化平方，不冒充电机能耗。
#   2. 首个合法回执只建立基准；重新验证输入并拒绝派生溢出，不虚构此前零速度。
# 输入：
#   previous：上一回执，首步为 None。
#   current：本步实际回执。
#   limits：速度变化采用的固定物理尺度。
# 输出：
#   cost：有限且非负的动作变化代价。
def transport_command_change_squared(previous: ControlApplicationRecord | None,
                                     current: ControlApplicationRecord, *,
                                     limits: PilotControlLimits) -> float:
    limits = _validated_limits(limits)
    if not isinstance(current, ControlApplicationRecord) or (
        previous is not None and not isinstance(previous, ControlApplicationRecord)
    ):
        raise ValueError("TRAINING_CONTROL_EVIDENCE_TYPE_INVALID")
    current = ControlApplicationRecord.model_validate(current.model_dump(mode="python"))
    if previous is None:
        cost = 0.
        return cost
    previous = ControlApplicationRecord.model_validate(previous.model_dump(mode="python"))
    scales = (limits.horizontal_speed_mps, limits.horizontal_speed_mps,
              limits.vertical_speed_mps)
    try:
        velocity_change = sum(((a-b)/scale)**2 for a, b, scale in zip(
            current.velocity_ned_mps, previous.velocity_ned_mps, scales, strict=True))
        # 先各自取模，不能先减去两个极大航向导致溢出或损失小角度变化。
        heading_change = (current.yaw_heading_deg % 360
                          - previous.yaw_heading_deg % 360 + 180) % 360 - 180
        cost = velocity_change + (heading_change / 180.)**2
    except OverflowError as error:
        raise ValueError("TRAINING_COMMAND_CHANGE_COST_NONFINITE") from error
    if not math.isfinite(cost):
        raise ValueError("TRAINING_COMMAND_CHANGE_COST_NONFINITE")
    return cost


# 功能：
#   1. 验证快照身份、特征期限、实际 NED 速度与偏航积分，恢复连续手柄四轴标签。
#   2. 明确区分源姿态下的教师模仿和执行姿态下的动作归因，拒绝越界标签而不裁剪。
#   3. 教师恢复仅接纳真实纯速度传输及安全层零偏航；历史位置恢复仍不能变成手柄标签。
# 输入：
#   snapshot：原始模型输入的严格 JSON 对象。
#   command：经批准的速度命令。
#   application：实际接受且含偏航积分的纯速度回执。
#   limits：与输入声明一致的部署尺度。
#   execution_frame：True 使用执行姿态，False 使用输入姿态。
# 输出：
#   control：前、右、上与顺时针偏航四个归一化控制幅度。
def executed_pilot_control(snapshot: dict, command: RuntimeLocalSafetyCommand,
                           application: ControlApplicationRecord, *, limits: PilotControlLimits,
                           execution_frame: bool = False) -> NormalizedPilotControl:
    if type(execution_frame) is not bool:
        raise ValueError("DEMONSTRATION_EXECUTION_FRAME_INVALID")
    if type(snapshot) is not dict:
        raise ValueError("DEMONSTRATION_SNAPSHOT_TYPE_INVALID")
    snapshot = copy_json(snapshot)
    limits = _validated_limits(limits)
    command, application = _control_evidence(command, application)
    content = dict(snapshot)
    digest = content.pop("snapshot_sha256", None)
    if digest != sha256_json(content):
        raise ValueError("DEMONSTRATION_SNAPSHOT_HASH_MISMATCH")
    if application.transport != "velocity-ned":
        raise ValueError("DEMONSTRATION_POSITION_CONTROL_IS_NOT_A_VELOCITY_LABEL")
    if application.yaw_rate_application is None:
        raise ValueError("DEMONSTRATION_EXECUTED_YAW_RATE_MISSING")
    accepted = application.accepted_at_unix_ms
    reference_time = snapshot.get("control_reference_observed_at_unix_ms")
    if (
        not isinstance(reference_time, int)
        or isinstance(reference_time, bool)
        or not reference_time
        <= command.generated_at_unix_ms
        <= accepted
        <= command.valid_until_unix_ms
        or accepted - reference_time > 250
    ):
        raise ValueError("DEMONSTRATION_CONTROL_TIME_ALIGNMENT_INVALID")
    features = RealtimeFeatureSnapshot.model_validate(snapshot.get("realtime_feature_snapshot"))
    if features.policy_feature_contract_sha256() != CURRENT_POLICY_FEATURE_CONTRACT_SHA256:
        raise ValueError("DEMONSTRATION_FEATURE_CONTRACT_MISMATCH")
    if not features.fresh_at(accepted):
        raise ValueError("DEMONSTRATION_CONTROL_INPUT_STALE")
    flight = next(
        (
            encoding
            for encoding in features.encodings
            if encoding.encoder_role == "flight-state-encoder"
        ),
        None,
    )
    if flight is None or flight.valid_mask[:4] != [1.0, 1.0, 1.0, 1.0]:
        raise ValueError("DEMONSTRATION_BODY_FRAME_UNAVAILABLE")
    if any(
        not 0 <= accepted - encoding.observed_at_unix_ms <= 250
        for encoding in features.encodings
        if encoding.encoder_role in features.required_roles
    ):
        raise ValueError("DEMONSTRATION_CONTROL_INPUT_STALE")
    teacher_velocity_recovery = (
        command.decision.action == "replan"
        and command.navigation_control_authority == "route-fallback"
        and not command.model_navigation_authorized
        and command.requested_control_intent is None
        and command.decision.control_source == "deterministic-brake"
        and command.decision.selected_yaw_rate_dps == 0.0
        and application.yaw_rate_application.clockwise_rate_dps == 0.0
    )
    if command.decision.action not in {"continue", "slow"} and not teacher_velocity_recovery:
        raise ValueError("DEMONSTRATION_SAFETY_OVERRIDE_IS_NOT_TEACHER_BEHAVIOR")
    yaw_rate = application.yaw_rate_application.clockwise_rate_dps
    approved_yaw = command.decision.selected_yaw_rate_dps
    if abs(yaw_rate) > abs(approved_yaw) + 1e-6 or yaw_rate * approved_yaw < -1e-6:
        raise ValueError("DEMONSTRATION_EXECUTED_YAW_EXCEEDS_APPROVAL")
    strategic = snapshot.get("strategic_context")
    task = strategic.get("task") if isinstance(strategic, dict) else None
    declared = task.get("normalized_pilot_control_limits") if isinstance(task, dict) else None
    if not isinstance(declared, dict) or any(
        declared.get(name) != value
        for name, value in (
            ("horizontal_speed_mps", limits.horizontal_speed_mps),
            ("vertical_speed_mps", limits.vertical_speed_mps),
            ("yaw_rate_dps", limits.yaw_rate_dps),
        )
    ):
        raise ValueError("DEMONSTRATION_DEPLOYMENT_LIMITS_MISMATCH")
    velocity = command.decision.selected_velocity_mps
    if any(
        abs(actual - expected) > 1e-6
        for actual, expected in zip(
            application.velocity_ned_mps,
            (velocity.y, velocity.x, -velocity.z),
            strict=True,
        )
    ):
        raise ValueError("DEMONSTRATION_EXECUTED_VELOCITY_DIFFERS_FROM_APPROVAL")
    intent = command.requested_control_intent
    if intent is not None and intent.navigation_snapshot_sha256 != digest:
        raise ValueError("DEMONSTRATION_WRONG_MODEL_OBSERVATION")
    orientation = QuaternionWxyz(
        w=flight.features[0], x=flight.features[1], y=flight.features[2], z=flight.features[3]
    )
    if execution_frame:
        # Native action/reward attribution must use the same rotation that
        # converted the joystick proposal. Re-projecting with the older input
        # attitude invents a safety intervention even on an unchanged action.
        # Source-frame teacher imitation is a separate explicit default above.
        if command.evaluated_body_orientation_world_from_body is None:
            raise ValueError("NATIVE_EXECUTION_BODY_FRAME_EVIDENCE_MISSING")
        orientation = command.evaluated_body_orientation_world_from_body
    north, east, down = application.velocity_ned_mps
    body = world_enu_to_body(orientation, Vector3(x=east, y=north, z=-down))
    # Reject out-of-envelope labels. Clipping would silently label a different
    # command from the one that was actually sent.
    control = NormalizedPilotControl(
        forward_axis=body.x / limits.horizontal_speed_mps,
        right_axis=body.y / limits.horizontal_speed_mps,
        up_axis=body.z / limits.vertical_speed_mps,
        yaw_axis=application.yaw_rate_application.clockwise_rate_dps / limits.yaw_rate_dps,
    )
    return control
