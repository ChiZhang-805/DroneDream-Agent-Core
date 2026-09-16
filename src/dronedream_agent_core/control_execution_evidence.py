"""Join actual transport acceptance to the original sensor/model authority.

Acceptance is not proof that the aircraft moved successfully. Physical outcome
comes from native telemetry and the independent mission verifier. These records
prove which bounded command was actually sent, instead of counting proposals.
"""

from __future__ import annotations

import sys
from typing import Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from .contracts import BodyFrameControlIntent, RuntimeLocalSafetyCommand, StrictModel
from .control_authority import integrate_model_yaw
from .control_timing import LOCAL_CONTROL_MAXIMUM_AGE_SECONDS
from .hashing import sha256_json
from .realtime_feature_encoders import RealtimeFeatureSnapshot
from .runtime_evidence import runtime_evidence_inventory_complete


class YawRateApplication(StrictModel):
    """Rate-to-heading conversion actually supplied to the FC transport.

    This is a command, not measured yaw response. An absolute-heading sender
    cannot invent a zero rate by omitting this evidence.
    """

    previous_heading_deg: float = Field(strict=True)
    clockwise_rate_dps: float = Field(ge=-180, le=180, strict=True)
    integration_seconds: float = Field(gt=0, le=0.25, strict=True)


class ControlApplicationRecord(StrictModel):
    """Actual NED transport receipt bound to one original command and observation.

    Preserve safety replacements and late acceptances for auditing. Construction
    alone does not qualify the receipt; verification checks provenance/time.
    """
    sequence: int = Field(ge=1, strict=True)
    accepted_at_unix_ms: int = Field(ge=0, strict=True)
    command_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    command_generated_at_unix_ms: int = Field(ge=0, strict=True)
    command_valid_until_unix_ms: int = Field(ge=0, strict=True)
    transport: Literal["velocity-ned", "position-velocity-ned"]
    velocity_ned_mps: tuple[float, float, float]
    position_ned_m: tuple[float, float, float] | None = None
    yaw_heading_deg: float = Field(strict=True)
    yaw_rate_application: YawRateApplication | None = None
    safety_action: Literal["continue", "slow", "hold", "replan"]
    control_source: str = Field(min_length=1, max_length=80)
    model_authorized: bool = Field(strict=True)
    intent: BodyFrameControlIntent | None

    # 功能：
    #   在 Pydantic 转换前校验三轴物理量，禁止字符串或布尔值伪装成实际运动数据。
    # 输入：
    #   cls：控制回执模型类。
    #   value：NED 三轴向量或未提供的位置。
    # 输出：
    #   value：通过严格数值检查的原始向量。
    @field_validator("velocity_ned_mps", "position_ned_m", mode="before")
    @classmethod
    def physical_vector_is_numeric(cls, value):
        if value is not None and (
            not isinstance(value, (tuple, list)) or len(value) != 3
            or any(type(v) not in (int, float) or not -sys.float_info.max <= v <= sys.float_info.max
                   for v in value)
        ):
            raise ValueError("CONTROL_APPLICATION_VECTOR_INVALID")
        return value

    # 功能：
    #   速度回执序列化时省略不存在的位置字段，保持既有速度回执摘要身份。
    # 输入：
    #   self：当前回执。
    #   handler：Pydantic 标准序列化处理器。
    # 输出：
    #   payload：兼容既有摘要格式的回执字典。
    @model_serializer(mode="wrap")
    def preserve_velocity_receipt_identity(self, handler):
        payload = handler(self)
        if self.position_ned_m is None:
            payload.pop("position_ned_m", None)
        return payload

    # 功能：
    #   检查运输模式与实际位置字段一致，并按控制器同一算法验证偏航积分。
    #   回执仅记录已发送命令，不据此证明飞机实际产生了对应运动。
    # 输入：
    #   self：已通过字段验证的回执。
    # 输出：
    #   self：通过传输及偏航一致性检查的回执。
    @model_validator(mode="after")
    def yaw_conversion_is_bound(self):
        if self.transport == "velocity-ned" and self.position_ned_m is not None:
            raise ValueError("VELOCITY_RECEIPT_CANNOT_CLAIM_POSITION_CONTROL")
        if self.transport == "position-velocity-ned" and self.position_ned_m is None:
            raise ValueError("POSITION_CONTROL_RECEIPT_REQUIRES_ACTUAL_SETPOINT")
        yaw = self.yaw_rate_application
        if yaw is not None:
            expected = integrate_model_yaw(
                previous_heading_deg=yaw.previous_heading_deg,
                requested_rate_dps=yaw.clockwise_rate_dps,
                maximum_rate_dps=180, step_seconds=yaw.integration_seconds,
            )
            # 分别规范到单周角度后比较，避免大数相减丢失真实的小偏航量。
            error = (self.yaw_heading_deg % 360 - expected + 180) % 360 - 180
            if abs(error) > 1e-6:
                raise ValueError("CONTROL_APPLICATION_YAW_CONVERSION_MISMATCH")
        return self


# 功能：
#   捕获真正传给飞控传输层的设定值并绑定原始命令，保留超时回执以便查错。
#   拷贝意图和偏航证据，不让后续可变对象修改追溯改变已记录动作，也不续期。
# 输入：
#   command：原始批准命令。
#   sequence：实际接受回执的序号，从 1 开始。
#   accepted_at_unix_ms：传输层实际接受时刻，UNIX 毫秒。
#   transport：所用 NED 传输模式。
#   velocity_ned_mps：实际发送的北、东、下三轴速度，单位米每秒。
#   yaw_heading_deg：实际发送的航向角，单位度。
#   yaw_rate_application：偏航速度到航向的实际积分参数。
#   position_ned_m：位置速度联合模式的实际 NED 位置，单位米。
# 输出：
#   record：拥有独立内容的实际传输回执。
def control_application_record(
    command: RuntimeLocalSafetyCommand,
    *,
    sequence: int,
    accepted_at_unix_ms: int,
    transport: str,
    velocity_ned_mps: tuple[float, float, float],
    yaw_heading_deg: float,
    yaw_rate_application: YawRateApplication | None = None,
    position_ned_m: tuple[float, float, float] | None = None,
) -> ControlApplicationRecord:
    # 模型实例可经嵌套修改或 model_copy 绕过赋值检查，回执边界重新拥有并校验它。
    command = RuntimeLocalSafetyCommand.model_validate(command.model_dump(mode="python"))
    if isinstance(yaw_rate_application, YawRateApplication):
        yaw_rate_application = yaw_rate_application.model_dump(mode="python")
    record = ControlApplicationRecord(
        sequence=sequence,
        accepted_at_unix_ms=accepted_at_unix_ms,
        command_sha256=sha256_json(command),
        observation_sha256=command.observation_sha256,
        command_generated_at_unix_ms=command.generated_at_unix_ms,
        command_valid_until_unix_ms=command.valid_until_unix_ms,
        transport=transport,
        velocity_ned_mps=velocity_ned_mps,
        position_ned_m=position_ned_m,
        yaw_heading_deg=yaw_heading_deg,
        yaw_rate_application=yaw_rate_application,
        safety_action=command.decision.action,
        control_source=command.decision.control_source,
        model_authorized=command.model_navigation_authorized,
        intent=command.requested_control_intent,
    )
    return record


# 功能：
#   为悬停和运动训练步骤统一校验回执与原始命令的精确绑定，拒绝被修改的模型实例。
# 输入：
#   command：训练步骤引用的原始命令。
#   application：实际运输回执。
# 输出：
#   None：不返回业务数据。
def validate_application_binding(
    command: RuntimeLocalSafetyCommand, application: ControlApplicationRecord,
) -> None:
    command = RuntimeLocalSafetyCommand.model_validate(command.model_dump(mode="python"))
    application = ControlApplicationRecord.model_validate(application.model_dump(mode="python"))
    if (
        application.command_sha256 != sha256_json(command)
        or application.observation_sha256 != command.observation_sha256
        or application.command_generated_at_unix_ms != command.generated_at_unix_ms
        or application.command_valid_until_unix_ms != command.valid_until_unix_ms
        or application.safety_action != command.decision.action
        or application.control_source != command.decision.control_source
        or application.intent != command.requested_control_intent
        or application.model_authorized != command.model_navigation_authorized
    ):
        raise ValueError("DEMONSTRATION_EXECUTED_COMMAND_MISMATCH")


# 功能：
#   1. 对照完整落盘清单，将实际回执连接到原始命令和带摘要的模型输入快照。
#   2. 验证指令归属、传输设定值、偏航授权和输入期限，拒绝缺失或损坏证据。
#   3. 分别统计模型控制和确定性安全替代；这是离线证据验收，不是飞行结果验收。
# 输入：
#   records：按接受顺序读取的控制回执。
#   navigation_snapshots：模型输入快照信封列表。
#   command_records：原始批准命令信封列表。
#   expected_count：预期回执条数。
#   writer_summary：证据写入器的完成汇总。
#   writer_artifact_counts：独立回读各证据文件后统计的条数。
#   application_artifact_path：控制回执文件在写入清单中的路径。
# 输出：
#   result：是否通过、拒绝代码、动作分类计数及最大模型输入年龄。
def verify_control_applications(
    records: list[dict],
    *,
    navigation_snapshots: list[dict],
    command_records: list[dict],
    expected_count: int,
    writer_summary: dict | None,
    writer_artifact_counts: dict[str, int],
    application_artifact_path: str,
) -> dict:
    issues = set()
    snapshots = {}
    commands = {}
    if not isinstance(records, list):
        issues.add("CONTROL_APPLICATION_EVIDENCE_INVALID")
        records = []
    if not isinstance(command_records, list):
        issues.add("CONTROL_APPLICATION_COMMAND_EVIDENCE_INVALID")
        command_records = []
    if not isinstance(navigation_snapshots, list):
        issues.add("CONTROL_APPLICATION_SNAPSHOT_EVIDENCE_INVALID")
        navigation_snapshots = []
    for envelope in command_records:
        if not isinstance(envelope, dict):
            issues.add("CONTROL_APPLICATION_COMMAND_EVIDENCE_INVALID")
            continue
        payload = envelope.get("command")
        # 感知可以真实完成而命令在发布前已过期；显式 null 表示没有授权，
        # 不是损坏的命令。它不进入摘要索引，任何声称执行了该命令的回执仍会失败。
        if "command" in envelope and payload is None:
            continue
        if not isinstance(payload, dict):
            issues.add("CONTROL_APPLICATION_COMMAND_EVIDENCE_INVALID")
            continue
        try:
            command = RuntimeLocalSafetyCommand.model_validate(payload)
            commands[sha256_json(command)] = command
        except (ValueError, TypeError, OverflowError, RecursionError):
            issues.add("CONTROL_APPLICATION_COMMAND_EVIDENCE_INVALID")
    for envelope in navigation_snapshots:
        if not isinstance(envelope, dict):
            issues.add("CONTROL_APPLICATION_SNAPSHOT_EVIDENCE_INVALID")
            continue
        payload = envelope.get("snapshot")
        if not isinstance(payload, dict):
            issues.add("CONTROL_APPLICATION_SNAPSHOT_EVIDENCE_INVALID")
            continue
        content = dict(payload)
        digest = content.pop("snapshot_sha256", None)
        try:
            if not isinstance(digest, str) or sha256_json(content) != digest:
                issues.add("CONTROL_APPLICATION_SNAPSHOT_EVIDENCE_INVALID")
                continue
            snapshots[digest] = payload
        except (ValueError, TypeError, OverflowError, RecursionError):
            issues.add("CONTROL_APPLICATION_SNAPSHOT_EVIDENCE_INVALID")
    if (type(expected_count) is not int or expected_count <= 0
            or not records or len(records) != expected_count):
        issues.add("CONTROL_APPLICATION_RECORD_COUNT_MISMATCH")
    # This FIFO owns both acceptance receipts and executor history. Compare
    # each independently recounted artifact, never their total to one stream.
    if (not isinstance(application_artifact_path, str) or not application_artifact_path
            or not isinstance(writer_artifact_counts, dict)
            or writer_artifact_counts.get(application_artifact_path) != len(records)
            or not runtime_evidence_inventory_complete(
                writer_summary, artifact_counts=writer_artifact_counts,
                record_count=sum(value for value in writer_artifact_counts.values()
                                 if type(value) is int))):
        issues.add("CONTROL_APPLICATION_EVIDENCE_NOT_DRAINED")
    motion_count, safety_motion_count, maximum_input_age_ms, latest = 0, 0, 0, -1
    for index, raw in enumerate(records, start=1):
        try:
            record = ControlApplicationRecord.model_validate(raw)
            if record.sequence != index or record.accepted_at_unix_ms < latest:
                issues.add("CONTROL_APPLICATION_ORDER_INVALID")
            latest = record.accepted_at_unix_ms
            command = commands.get(record.command_sha256)
            if command is None:
                issues.add("CONTROL_APPLICATION_COMMAND_MISSING")
                continue
            if (
                record.observation_sha256 != command.observation_sha256
                or record.intent != command.requested_control_intent
                or record.model_authorized != command.model_navigation_authorized
                or record.safety_action != command.decision.action
                or record.control_source != command.decision.control_source
                or record.command_generated_at_unix_ms != command.generated_at_unix_ms
                or record.command_valid_until_unix_ms != command.valid_until_unix_ms
            ):
                issues.add("CONTROL_APPLICATION_COMMAND_BINDING_MISMATCH")
                continue
            intent = record.intent
            if record.safety_action in {"continue", "slow", "replan"} and not (
                record.command_generated_at_unix_ms <= record.accepted_at_unix_ms
                <= record.command_valid_until_unix_ms
            ):
                # Teachers are not exempt from the student's physical input
                # lease. A completed route alone cannot bless late actuation.
                issues.add("CONTROL_APPLICATION_TRANSPORT_DEADLINE_VIOLATION")
            safety_motion = record.control_source == "deterministic-safety-override"
            if intent is None or (record.control_source != "local-model-body-control"
                                  and not safety_motion):
                continue  # Safety/hold has its own attribution, not model motion.
            if record.safety_action not in {"continue", "slow"}:
                continue
            if safety_motion:
                safety_motion_count += 1
            else:
                motion_count += 1
            selected = command.decision.selected_velocity_mps
            if any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(
                    record.velocity_ned_mps, (selected.y, selected.x, -selected.z), strict=True
                )
            ):
                issues.add("CONTROL_APPLICATION_VELOCITY_DIFFERS_FROM_APPROVED_COMMAND")
            if (
                record.transport != "velocity-ned"
                or not record.model_authorized
                or intent.control_origin != "continuous-model-output"
                or (not safety_motion and intent.yaw_control_mode != "model-rate")
            ):
                issues.add("CONTROL_APPLICATION_SAFETY_VELOCITY_TRANSPORT_INVALID"
                           if safety_motion else "CONTROL_APPLICATION_NOT_DIRECT_MODEL_CONTROL")
            yaw = record.yaw_rate_application
            selected_rate = command.decision.selected_yaw_rate_dps
            if (
                yaw is None
                or abs(yaw.clockwise_rate_dps) > abs(selected_rate) + 1e-6
                or yaw.clockwise_rate_dps * selected_rate < -1e-6
                or (safety_motion and (selected_rate != 0 or yaw.clockwise_rate_dps != 0))
            ):
                issues.add("CONTROL_APPLICATION_YAW_AUTHORITY_MISMATCH")
            if not (
                intent.generated_at_unix_ms
                <= record.command_generated_at_unix_ms
                <= record.accepted_at_unix_ms
                <= record.command_valid_until_unix_ms
                <= intent.valid_until_unix_ms
            ):
                issues.add("CONTROL_APPLICATION_AUTHORITY_EXPIRED")
            snapshot = snapshots.get(intent.navigation_snapshot_sha256)
            if snapshot is None:
                issues.add("CONTROL_APPLICATION_MODEL_INPUT_MISSING")
                continue
            features = RealtimeFeatureSnapshot.model_validate(snapshot["realtime_feature_snapshot"])
            if not features.fresh_at(record.accepted_at_unix_ms):
                issues.add("CONTROL_APPLICATION_SENSOR_EVIDENCE_STALE")
            source_time = min(
                e.observed_at_unix_ms
                for e in features.encodings
                if e.encoder_role in features.required_roles
            )
            age = record.accepted_at_unix_ms - source_time
            maximum_input_age_ms = max(maximum_input_age_ms, age)
            if not 0 <= age <= int(LOCAL_CONTROL_MAXIMUM_AGE_SECONDS * 1000):
                issues.add("CONTROL_APPLICATION_INPUT_DEADLINE_VIOLATION")
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            issues.add("CONTROL_APPLICATION_EVIDENCE_INVALID")
    if motion_count == 0:
        issues.add("CONTROL_APPLICATION_HAS_NO_MODEL_MOTION")
    result = {
        "accepted": not issues,
        "issue_codes": sorted(issues),
        "record_count": len(records),
        "model_motion_record_count": motion_count,
        "safety_velocity_record_count": safety_motion_count,
        "maximum_model_input_age_ms": maximum_input_age_ms,
    }
    return result
