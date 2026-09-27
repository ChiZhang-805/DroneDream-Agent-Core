"""In-process simulation-teacher checkpoint receipts, never model approvals."""

import math
from typing import Literal

from pydantic import Field

from ..contracts import (
    RuntimeAssessment,
    RuntimeCheckpointContract,
    RuntimeCheckpointRequest,
    StrictModel,
)
from ..hashing import sha256_json


class PayloadTeacherCheckpointDecision(StrictModel):
    schema_version: Literal['dronedream.payload-teacher-checkpoint.v1'] = (
        'dronedream.payload-teacher-checkpoint.v1')
    request_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    checkpoint_contract_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    assessment: RuntimeAssessment
    continue_authorized: bool
    simulation_only: Literal[True] = True
    model_call_performed: Literal[False] = False
    flight_qualification_granted: Literal[False] = False


# 功能：
#   1. 为显式负载教师课程复核刚产生的检查点请求，不读取或伪造云端模型调用。
#   2. 重新计算位置、速度和电池门槛；请求不属于冻结契约或派生数值不一致时拒绝。
# 输入：
#   request：执行器在稳定检查点内刚取得的原生遥测请求，不是外部待处理文件。
#   contract：本次仿真冻结的检查点契约。
# 输出：
#   decision：单独类型的教师回执，不能被正式模型检查点校验器接受。
def decide_payload_teacher_checkpoint(request, contract):
    request = RuntimeCheckpointRequest.model_validate(request.model_dump(), strict=True)
    contract = RuntimeCheckpointContract.model_validate(contract.model_dump(), strict=True)
    matches = [item for item in contract.checkpoints
               if item.checkpoint_id == request.checkpoint.checkpoint_id]
    if (request.contract_id != contract.contract_id or len(matches) != 1
            or matches[0] != request.checkpoint):
        raise ValueError('PAYLOAD_TEACHER_CHECKPOINT_BINDING_INVALID')
    observed, commanded = request.observed_position_ned_m, request.commanded_position_ned_m
    position_error = math.dist((observed.x, observed.y, observed.z),
                               (commanded.x, commanded.y, commanded.z))
    velocity = request.observed_velocity_ned_mps
    speed = math.hypot(velocity.x, velocity.y, velocity.z)
    if (not math.isclose(position_error, request.position_error_m, abs_tol=1e-6)
            or not math.isclose(speed, request.speed_mps, abs_tol=1e-6)):
        raise ValueError('PAYLOAD_TEACHER_CHECKPOINT_TELEMETRY_MISMATCH')
    gates = {'position_error_within_0_75_m': position_error <= .75,
             'speed_within_0_50_mps': speed <= .5,
             'battery_above_10_percent': request.battery_percent > 10.,
             'telemetry_finite': True}
    if request.deterministic_gates != gates:
        raise ValueError('PAYLOAD_TEACHER_CHECKPOINT_GATES_MISMATCH')
    accepted = all(gates.values())
    decision = PayloadTeacherCheckpointDecision(
        request_sha256=sha256_json(request), checkpoint_contract_sha256=sha256_json(contract),
        assessment=RuntimeAssessment(action='accept' if accepted else 'abort',
            issue_codes=[] if accepted else ['PAYLOAD_TEACHER_CHECKPOINT_REJECTED']),
        continue_authorized=accepted)
    return decision
