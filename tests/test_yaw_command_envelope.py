"""Command shaping happens before collision arbitration; not a physical stability proof."""

import pytest

from dronedream_agent_core.contracts import BodyFrameControlIntent, QuaternionWxyz, RuntimeLocalSafetyObservation, Vector3
from dronedream_agent_core.yaw_command_envelope import YawCommandEnvelope, shape_model_yaw
from dronedream_agent_core.runtime_local_safety import evaluate_runtime_local_safety
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord, YawRateApplication
from test_runtime_local_safety import _vehicle


# 功能：
#   构建绑定当前任务和快照的测试意图与显式偏航配置。
# 输入：
#   requested：测试请求的顺时针角速度。
# 输出：
#   intent、envelope：独立的模型请求和安全裁决前包络。
def inputs(requested=20.):
    intent = BodyFrameControlIntent(source_expert='precision-maneuver-policy', control_origin='continuous-model-output',
        model_call_id='model-'+'a'*24, navigation_snapshot_sha256='b'*64, task_reference_sha256='c'*64,
        generated_at_unix_ms=1000, valid_until_unix_ms=1200, forward_velocity_mps=0., right_velocity_mps=0.,
        up_velocity_mps=0., yaw_rate_dps=requested, maximum_acceleration_mps2=2., maximum_jerk_mps3=10.)
    envelope = YawCommandEnvelope(navigation_snapshot_sha256='b'*64, task_reference_sha256='c'*64,
        source_observed_at_unix_ms=1000, measured_clockwise_heading_rate_dps=0., maximum_rate_dps=20.,
        maximum_command_acceleration_dps2=40., first_step_seconds=.05)
    return intent, envelope


# 功能：
#   检查普通指令限幅具有明确方向和物理时间单位，不修改原请求。
# 输入：
#   requested、expected：测试请求与受限输出。
# 输出：
#   None：每个方向仅在配置的角速度变化包络内变化。
@pytest.mark.parametrize('requested,expected', [(20., 2.), (-20., -2.), (1., 1.), (0., 0.)])
def test_yaw_request_is_rate_limited(requested, expected):
    intent, envelope = inputs(requested)
    assert shape_model_yaw(intent, envelope, now_unix_ms=1010) == pytest.approx(expected)
    assert intent.yaw_rate_dps == requested


# 功能：
#   不允许跨任务、过期来源或未来来源被包装成新的平滑指令。
# 输入：
#   damage：损坏的身份或时序。
# 输出：
#   None：非法包络在碰撞裁决之前明确拒绝。
@pytest.mark.parametrize('damage', ['task', 'snapshot', 'future', 'stale', 'expired'])
def test_yaw_envelope_rejects_invalid_binding(damage):
    intent, envelope = inputs()
    values = envelope.model_dump(mode='python')
    if damage == 'task':
        values['task_reference_sha256'] = 'd'*64
    elif damage == 'snapshot':
        values['navigation_snapshot_sha256'] = 'd'*64
    elif damage == 'future':
        values['source_observed_at_unix_ms'] = 1100
    elif damage == 'stale':
        values['source_observed_at_unix_ms'] = 700
    with pytest.raises(ValueError, match='YAW_ENVELOPE'):
        shape_model_yaw(intent, YawCommandEnvelope(**values), now_unix_ms=1200 if damage == 'expired' else 1010)


# 功能：
#   验证新包络通过现有安全入口，传感器不健康时最终零偏航不会被后置平滑改变。
# 输入：
#   healthy：测试感知健康状态。
# 输出：
#   None：健康请求受约束，失效请求仍由安全层制动。
@pytest.mark.parametrize('healthy', [True, False])
def test_envelope_precedes_final_safety_arbitration(healthy):
    intent, envelope = inputs()
    observation = RuntimeLocalSafetyObservation(sequence=1, observed_at_unix_ms=1000, source='onboard',
        stream_healthy=healthy, stream_age_seconds=.01, localization_covariance_m2=.01,
        current_position_m=Vector3(x=0., y=0., z=1.5), current_velocity_mps=Vector3(x=0., y=0., z=0.),
        target_position_m=Vector3(x=3., y=0., z=1.5), body_orientation_world_from_body=QuaternionWxyz(w=1., x=0., y=0., z=0.))
    command = evaluate_runtime_local_safety(observation=observation, vehicle=_vehicle(), static_primitives=[],
        required_clearance_m=.2, generated_at_unix_ms=1010, navigation_control_authority='model-required',
        navigation_goal_id='test-goal',
        model_navigation_authorized=True, model_call_id=intent.model_call_id, model_path_sha256=intent.task_reference_sha256,
        model_navigation_snapshot_sha256=intent.navigation_snapshot_sha256, requested_control_intent=intent, yaw_envelope=envelope)
    if healthy:
        assert command.decision.selected_yaw_rate_dps == pytest.approx(2.)
    else:
        assert command.decision.selected_yaw_rate_dps == 0.
        assert command.decision.action in ('hold', 'replan')


# 功能：
#   检验变化限制读取实际接受的偏航速度，而非未执行意图，并拒绝过期或跨任务回执。
# 输入：
#   damage：正常回执或单项损坏条件。
# 输出：
#   None：合法动作从实际八度每秒继续变化，非法历史明确拒绝。
@pytest.mark.parametrize('damage', ['none', 'task', 'future', 'expired', 'limit', 'unauthorized'])
def test_yaw_envelope_uses_actual_past_receipt(damage):
    intent, envelope = inputs()
    previous_intent = intent.model_copy(update={'generated_at_unix_ms': 940, 'valid_until_unix_ms': 990})
    rate = 21. if damage == 'limit' else 8.
    receipt = ControlApplicationRecord(sequence=1, accepted_at_unix_ms=970,
        command_sha256='d'*64, observation_sha256='e'*64, command_generated_at_unix_ms=950,
        command_valid_until_unix_ms=990, transport='velocity-ned', velocity_ned_mps=(0., 0., 0.),
        yaw_heading_deg=rate*.05, yaw_rate_application=YawRateApplication(previous_heading_deg=0.,
            clockwise_rate_dps=rate, integration_seconds=.05), safety_action='continue', control_source='test',
        model_authorized=True, intent=previous_intent)
    if damage == 'task':
        receipt.intent = previous_intent.model_copy(update={'task_reference_sha256': 'f'*64})
    elif damage == 'future':
        receipt.accepted_at_unix_ms = 1011
    elif damage == 'expired':
        receipt.accepted_at_unix_ms = 990
    elif damage == 'unauthorized':
        receipt.model_authorized = False
    envelope.previous_application = receipt
    if damage == 'none':
        assert shape_model_yaw(intent, envelope, now_unix_ms=1010) == pytest.approx(9.6)
    else:
        with pytest.raises(ValueError, match='YAW_ENVELOPE'):
            shape_model_yaw(intent, envelope, now_unix_ms=1010)
