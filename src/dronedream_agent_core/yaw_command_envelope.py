"""Source-bound command shaping before safety arbitration, never after it."""

import math

from pydantic import Field

from .contracts import BodyFrameControlIntent, StrictModel
from .control_execution_evidence import ControlApplicationRecord


class YawCommandEnvelope(StrictModel):
    """Explicit configured command-rate bounds; not a claim about physical yaw response."""

    navigation_snapshot_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    task_reference_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_observed_at_unix_ms: int = Field(ge=0, strict=True)
    measured_clockwise_heading_rate_dps: float = Field(ge=-720, le=720, strict=True)
    maximum_rate_dps: float = Field(gt=0, le=180, strict=True)
    maximum_command_acceleration_dps2: float = Field(gt=0, le=1440, strict=True)
    first_step_seconds: float = Field(gt=0, le=.25, strict=True)
    previous_application: ControlApplicationRecord | None = None


# 功能：
#   在安全裁决前限制普通模型偏航请求的变化，参考过去实际接受值而非上一轮未执行提案。
#   后续碰撞、失效或紧急制动裁决可以覆盖此舒适性约束，禁止对安全结果再次平滑。
# 输入：
#   intent：绑定当前任务和快照的原模型请求；envelope：显式机型配置、同源转速及可选实际回执。
#   now_unix_ms：当前裁决时间，不刷新历史或观测时间。
# 输出：
#   requested_rate_dps：交给完整安全裁决的偏航候选，不是直接发往飞控的命令。
def shape_model_yaw(intent: BodyFrameControlIntent, envelope: YawCommandEnvelope, *, now_unix_ms: int) -> float:
    if not isinstance(intent, BodyFrameControlIntent) or not isinstance(envelope, YawCommandEnvelope):
        raise ValueError('YAW_ENVELOPE_INPUT_TYPE_INVALID')
    intent = BodyFrameControlIntent.model_validate(intent.model_dump(mode='python'), strict=True)
    envelope = YawCommandEnvelope.model_validate(envelope.model_dump(mode='python'), strict=True)
    if (type(now_unix_ms) is not int or not intent.generated_at_unix_ms <= now_unix_ms < intent.valid_until_unix_ms
            or not envelope.source_observed_at_unix_ms <= now_unix_ms <= envelope.source_observed_at_unix_ms+250
            or envelope.navigation_snapshot_sha256 != intent.navigation_snapshot_sha256
            or envelope.task_reference_sha256 != intent.task_reference_sha256
            or intent.control_origin != 'continuous-model-output' or intent.yaw_control_mode != 'model-rate'):
        raise ValueError('YAW_ENVELOPE_SOURCE_OR_AUTHORITY_MISMATCH')
    reference = max(-envelope.maximum_rate_dps, min(envelope.maximum_rate_dps, envelope.measured_clockwise_heading_rate_dps))
    step = envelope.first_step_seconds
    previous = envelope.previous_application
    if previous is not None:
        if (previous.transport != 'velocity-ned' or previous.yaw_rate_application is None
                or previous.intent is None or previous.intent.task_reference_sha256 != intent.task_reference_sha256
                or not previous.model_authorized
                or not previous.command_generated_at_unix_ms <= previous.accepted_at_unix_ms < previous.command_valid_until_unix_ms
                or not 0 < now_unix_ms-previous.accepted_at_unix_ms <= 250):
            raise ValueError('YAW_ENVELOPE_PREVIOUS_RECEIPT_INVALID')
        reference = previous.yaw_rate_application.clockwise_rate_dps
        if abs(reference) > envelope.maximum_rate_dps:
            raise ValueError('YAW_ENVELOPE_PREVIOUS_LIMIT_MISMATCH')
        step = (now_unix_ms-previous.accepted_at_unix_ms)/1000.
    target = max(-envelope.maximum_rate_dps, min(envelope.maximum_rate_dps, intent.yaw_rate_dps))
    delta = envelope.maximum_command_acceleration_dps2*step
    requested_rate_dps = max(reference-delta, min(reference+delta, target))
    if not math.isfinite(requested_rate_dps):
        raise ValueError('YAW_ENVELOPE_NONFINITE_OUTPUT')
    return requested_rate_dps
