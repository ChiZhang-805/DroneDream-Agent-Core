"""Teacher measurements cannot masquerade as model checkpoint approval."""

import pytest

from dronedream_agent_core.contracts import (
    RuntimeCheckpoint,
    RuntimeCheckpointContract,
    RuntimeCheckpointDecision,
    RuntimeCheckpointRequest,
    Vector3,
)
from dronedream_agent_core.training.payload_checkpoint import decide_payload_teacher_checkpoint


# 功能：
#   生成有限静止遥测与同一任务检查点，供无飞行副作用的边界测试。
# 输入：
#   无。
# 输出：
#   request、contract：绑定同一任务的测试请求和契约。
def checkpoint_fixture():
    checkpoint = RuntimeCheckpoint(checkpoint_id='checkpoint-001', segment_id='segment-001',
                                   task_id='arrive', track_point_index=1, target_node='pickup')
    contract = RuntimeCheckpointContract(contract_id='teacher-test', checkpoints=[checkpoint])
    request = RuntimeCheckpointRequest(contract_id=contract.contract_id, checkpoint=checkpoint,
        observed_position_ned_m=Vector3(x=0., y=0., z=-2.),
        commanded_position_ned_m=Vector3(x=0., y=0., z=-2.),
        observed_velocity_ned_mps=Vector3(x=0., y=0., z=0.),
        position_error_m=0., speed_mps=0., battery_percent=80., deterministic_gates={
            'position_error_within_0_75_m': True, 'speed_within_0_50_mps': True,
            'battery_above_10_percent': True, 'telemetry_finite': True})
    return request, contract


# 功能：
#   验证教师回执明确标记无模型调用，正式模型检查点结构拒绝此回执。
# 输入：
#   无。
# 输出：
#   None：两类资格不能混用，否则测试失败。
def test_teacher_receipt_is_not_model_evidence():
    request, contract = checkpoint_fixture()
    result = decide_payload_teacher_checkpoint(request, contract)
    assert result.continue_authorized and not result.model_call_performed
    assert not result.flight_qualification_granted
    with pytest.raises(ValueError):
        RuntimeCheckpointDecision.model_validate(result.model_dump())


# 功能：
#   验证契约错配、遥测派生值被改动或门槛缺失时拒绝，即使记录声称全部正常。
# 输入：
#   damage：需要注入的错误。
# 输出：
#   None：损坏请求无法获得教师授权，否则测试失败。
@pytest.mark.parametrize('damage', [
    'contract', 'checkpoint', 'position', 'speed', 'gate', 'duplicate'])
def test_teacher_recomputes_binding_and_gates(damage):
    request, contract = checkpoint_fixture()
    if damage == 'contract':
        request.contract_id = 'other'
    elif damage == 'checkpoint':
        request.checkpoint = request.checkpoint.model_copy(update={'task_id': 'other'})
    elif damage == 'position':
        request.position_error_m = .3
    elif damage == 'speed':
        request.speed_mps = .3
    elif damage == 'gate':
        request.deterministic_gates.pop('telemetry_finite')
    else:
        contract.checkpoints.append(contract.checkpoints[0].model_copy())
    with pytest.raises(ValueError):
        decide_payload_teacher_checkpoint(request, contract)


# 功能：
#   检查原生电量触及边界时返回拒绝，不能因教师课程而放宽停止条件。
# 输入：
#   无。
# 输出：
#   None：回执明确禁止继续，否则测试失败。
def test_low_battery_aborts_teacher_checkpoint():
    request, contract = checkpoint_fixture()
    request.battery_percent = 10.
    request.deterministic_gates['battery_above_10_percent'] = False
    decision = decide_payload_teacher_checkpoint(request, contract)
    assert not decision.continue_authorized and decision.assessment.action == 'abort'
