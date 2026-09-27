"""Synthetic transport fixtures; these tests never qualify a demonstration flight."""

import copy
import math
from contextlib import nullcontext

import pytest
from control_fixtures import complete_feature_snapshot

from dronedream_agent_core.contracts import (
    PredictiveSafetyDecision,
    QuaternionWxyz,
    RuntimeLocalSafetyCommand,
    Vector3,
)
from dronedream_agent_core.control_execution_evidence import control_application_record
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.pilot_control_mapping import PilotControlLimits
from dronedream_agent_core.realtime_feature_encoders import body_to_world_enu
from dronedream_agent_core.training.executed_control import (
    executed_pilot_control,
    executed_training_action,
    transport_command_change_squared,
)
from scripts.build_local_policy_dataset import _teacher_control_label


# 功能：
#   构造具有实际速度传输回执的教师样本，不用规划器建议或实测飞行速度冒充控制标签。
# 输入：
#   无。
# 输出：
#   snapshot：原始模型输入快照。
#   command：经批准的教师命令。
#   application：实际接受的传输回执。
#   limits：部署时采用的速度及偏航归一化尺度。
def teacher_evidence():
    limits = PilotControlLimits(0.8, 0.75, 20.0)
    snapshot = {
        "current_position_m": {"x": 0.0, "y": 0.0, "z": 1.0},
        "goal_position_m": {"x": 9.0, "y": 7.0, "z": 2.0},
        "control_reference_observed_at_unix_ms": 1000,
        "strategic_context": {
            "task": {
                "local_navigation_output_mode": "normalized-body-velocity",
                "control_profile": "cruise",
                "normalized_pilot_control_limits": {
                    "horizontal_speed_mps": 0.8,
                    "vertical_speed_mps": 0.75,
                    "yaw_rate_dps": 20.0,
                },
            }
        },
        "realtime_feature_snapshot": complete_feature_snapshot().model_dump(mode="json"),
    }
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    command = RuntimeLocalSafetyCommand(
        observation_sha256="c" * 64,
        observation_sequence=1,
        generated_at_unix_ms=1020,
        valid_until_unix_ms=1200,
        source="onboard",
        evaluated_body_orientation_world_from_body=QuaternionWxyz(w=1, x=0, y=0, z=0),
        evaluated_target_position_m=Vector3(x=1, y=2, z=3),
        navigation_goal_id="office",
        navigation_control_authority="route-fallback",
        command_position_m=Vector3(x=1, y=2, z=3),
        decision=PredictiveSafetyDecision(
            action="continue",
            selected_velocity_mps=Vector3(x=0.4, y=-0.2, z=0.375),
            selected_yaw_rate_dps=10.0,
            control_source="route-target",
            predicted_path_m=[Vector3(x=1, y=2, z=3)],
            minimum_predicted_clearance_m=1,
            time_to_minimum_clearance_seconds=0.1,
            evaluated_candidate_count=1,
        ),
    )
    application = control_application_record(
        command,
        sequence=1,
        accepted_at_unix_ms=1050,
        transport="velocity-ned",
        velocity_ned_mps=(-0.2, 0.4, -0.375),
        yaw_heading_deg=20.5,
        yaw_rate_application={
            "previous_heading_deg": 20.0,
            "clockwise_rate_dps": 10.0,
            "integration_seconds": 0.05,
        },
    )
    return snapshot, command, application, limits


# 功能：
#   验证标签来自实际接受的机体控制轴，并且不以教师私有短目标改写学生输入。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_demonstration_uses_actual_body_axes_and_preserves_original_observation():
    snapshot, command, application, limits = teacher_evidence()
    original = copy.deepcopy(snapshot)
    target = executed_pilot_control(snapshot, command, application, limits=limits)
    assert [target.forward_axis, target.right_axis, target.up_axis, target.yaw_axis] == (
        pytest.approx([0.5, 0.25, 0.5, 0.5])
    )
    assert snapshot == original
    # A teacher's private short lookahead is irrelevant to the student's input.
    label = _teacher_control_label(
        snapshot,
        {
            "command": command.model_dump(mode="json"),
            "route_target_m": {"x": 0.1, "y": 0, "z": 1},
        },
        application=application,
        route_speed_limit_mps=0.8,
    )
    assert label[2] == pytest.approx([0.5, 0.25, 0.5, 0.5])
    assert snapshot == original


# 功能：
#   验证训练奖惩使用执行瞬间的姿态还原控制轴，不借用更早输入的姿态制造干预。
# 输入：
#   rotation_axis：发生姿态变化的坐标轴。
#   angle：旋转角，单位弧度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("rotation_axis", ["x", "y", "z"])
@pytest.mark.parametrize("angle", [-.5, .5])
def test_native_applied_axes_use_exact_execution_rotation_not_old_input(rotation_axis, angle):
    snapshot, command, application, limits = teacher_evidence()
    quaternion = dict(w=math.cos(angle / 2), x=0., y=0., z=0.)
    quaternion[rotation_axis] = math.sin(angle / 2)
    orientation = QuaternionWxyz(**quaternion)
    velocity = body_to_world_enu(orientation, Vector3(x=.4, y=.2, z=.375))
    command = command.model_copy(update={
        "evaluated_body_orientation_world_from_body": orientation,
        "decision": command.decision.model_copy(update={"selected_velocity_mps": velocity}),
    })
    application = control_application_record(command, sequence=1, accepted_at_unix_ms=1050,
        transport="velocity-ned", velocity_ned_mps=(velocity.y, velocity.x, -velocity.z),
        yaw_heading_deg=application.yaw_heading_deg,
        yaw_rate_application=application.yaw_rate_application)
    original = copy.deepcopy(snapshot)
    applied, intervened = executed_training_action(snapshot, command, application, limits=limits)
    assert applied.axes == pytest.approx([.5, .25, .5, .5], abs=1e-9)
    assert not intervened and snapshot == original


# 功能：
#   验证执行姿态缺失时拒绝奖惩归因，历史教师的源姿态投影仍是明确区分的操作。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_missing_execution_rotation_is_not_silently_borrowed_from_old_model_input():
    snapshot, command, application, limits = teacher_evidence()
    command = command.model_copy(update={"evaluated_body_orientation_world_from_body": None})
    legacy_bytes = command.model_dump_json()
    assert "evaluated_body_orientation_world_from_body" not in legacy_bytes
    restored = RuntimeLocalSafetyCommand.model_validate_json(legacy_bytes)
    assert restored.model_dump_json() == legacy_bytes
    application = application.model_copy(update={"command_sha256": sha256_json(command)})
    with pytest.raises(ValueError, match="EXECUTION_BODY_FRAME_EVIDENCE_MISSING"):
        executed_training_action(snapshot, command, application, limits=limits)
    # Source-frame projection of a historical teacher remains an explicitly
    # different operation, not proof of a student's unmodified executed axes.
    assert executed_pilot_control(snapshot, command, application, limits=limits).forward_axis == .5


# 功能：
#   验证过期运动被拒绝，而实际安全悬停被记录为干预且不续期模型控制。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_training_late_motion_remains_forbidden_but_actual_safety_hold_is_intervention():
    snapshot, command, application, limits = teacher_evidence()
    actual, intervened = executed_training_action(snapshot, command, application, limits=limits)
    assert actual.mode == "pilot-control" and not intervened
    assert actual.axes == pytest.approx([.5, .25, .5, .5])
    assert len(actual.axes) == 4  # The normalized-control schema string is not a fifth axis.
    late = application.model_copy(update={"accepted_at_unix_ms": 1201})
    with pytest.raises(ValueError, match="OUTSIDE_COMMAND_LIFETIME"):
        executed_training_action(snapshot, command, late, limits=limits)
    command = command.model_copy(update={"decision": command.decision.model_copy(update={
        "action": "hold", "control_source": "deterministic-brake",
        "selected_velocity_mps": Vector3(x=0, y=0, z=0), "selected_yaw_rate_dps": 0.,
    })})
    held = control_application_record(command, sequence=1, accepted_at_unix_ms=1201,
        transport="position-velocity-ned", velocity_ned_mps=(0., 0., 0.),
        position_ned_m=(1., 2., 3.), yaw_heading_deg=20.)
    action, intervened = executed_training_action(snapshot, command, held, limits=limits)
    assert action.mode == "hold" and action.axes == [0.] * 4 and intervened
    timely = held.model_copy(update={"accepted_at_unix_ms": 1199})
    assert not executed_training_action(snapshot, command, timely, limits=limits)[1]
    with pytest.raises(ValueError, match="PRECEDES_COMMAND"):
        executed_training_action(snapshot, command,
            held.model_copy(update={"accepted_at_unix_ms": 1019}), limits=limits)
    for update, issue in (
        ({"velocity_ned_mps": (.1, 0., 0.)}, "HOLD_RECEIPT_NOT_VERIFIABLE"),
        ({"position_ned_m": None}, "POSITION_CONTROL_RECEIPT_REQUIRES_ACTUAL_SETPOINT"),
    ):
        # 缺失位置现在在共享回执边界就拒绝；非零悬停速度仍由训练语义门槛拒绝。
        with pytest.raises(ValueError, match=issue):
            executed_training_action(snapshot, command,
                                     held.model_copy(update=update), limits=limits)


# 功能：
#   验证重规划保留实际位置及速度前馈，不反推不存在的手柄控制轴。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_replan_preserves_position_and_velocity_without_forging_joystick_axes():
    snapshot, command, _, limits = teacher_evidence()
    command = command.model_copy(update={"decision": command.decision.model_copy(update={
        "action": "replan", "control_source": "deterministic-brake",
        "selected_yaw_rate_dps": 0.,
    })})
    application = control_application_record(command, sequence=1, accepted_at_unix_ms=1100,
        transport="position-velocity-ned", velocity_ned_mps=(-.2, .4, -.375),
        position_ned_m=(1., 2., 3.), yaw_heading_deg=20.)
    applied, expired = executed_training_action(snapshot, command, application, limits=limits)
    assert not expired and applied.mode == "safety-position-control"
    assert applied.position_ned_m == (1., 2., 3.)
    assert applied.velocity_feedforward_ned_mps == (-.2, .4, -.375)
    assert "axes" not in applied.model_dump()
    for update, issue in (
        ({"accepted_at_unix_ms": 1201}, "REPLAN_RECEIPT_NOT_VERIFIABLE"),
        ({"velocity_ned_mps": (0., 0., 0.)}, "REPLAN_RECEIPT_NOT_VERIFIABLE"),
        ({"position_ned_m": None}, "POSITION_CONTROL_RECEIPT_REQUIRES_ACTUAL_SETPOINT"),
    ):
        with pytest.raises(ValueError, match=issue):
            executed_training_action(snapshot, command,
                                     application.model_copy(update=update), limits=limits)


# 功能：
#   核验新教师恢复从真实纯速度回执还原四轴，不给模型冒用恢复权限或添加未经批准的偏航。
# 输入：
#   source、authorized、rate：教师来源、模型授权与偏航的边界组合。
# 输出：
#   None：唯一合法组合恢复实际轴，其余组合均被拒绝。
@pytest.mark.parametrize('source,authorized,rate', [
    ('deterministic-brake', False, 0.), ('route-target', False, 0.),
    ('deterministic-brake', True, 0.), ('deterministic-brake', False, 1.),
])
def test_velocity_recovery_requires_teacher_safety_origin(source, authorized, rate):
    snapshot, command, _, limits = teacher_evidence()
    command = command.model_copy(update={'model_navigation_authorized': authorized,
        'decision': command.decision.model_copy(update={'action': 'replan',
            'control_source': source, 'selected_yaw_rate_dps': rate})})
    # 先构造实际被传输层记录的零／非零偏航，不通过篡改回执绕过命令绑定。
    expected_error = (pytest.raises(ValueError, match='route-fallback control cannot claim model authorization')
                      if authorized else nullcontext())
    with expected_error:
        application = control_application_record(command, sequence=1, accepted_at_unix_ms=1050,
            transport='velocity-ned', velocity_ned_mps=(-.2, .4, -.375),
            yaw_heading_deg=20. + rate * .05,
            yaw_rate_application={'previous_heading_deg': 20., 'clockwise_rate_dps': rate,
                                  'integration_seconds': .05})
    if authorized:
        return
    if source != 'deterministic-brake' or authorized or rate != 0.:
        with pytest.raises(ValueError, match='SAFETY_OVERRIDE_IS_NOT_TEACHER_BEHAVIOR'):
            executed_pilot_control(snapshot, command, application, limits=limits)
    else:
        target = executed_pilot_control(snapshot, command, application, limits=limits)
        assert [target.forward_axis, target.right_axis, target.up_axis, target.yaw_axis] == pytest.approx([.5, .25, .5, 0.])
        applied, expired = executed_training_action(snapshot, command, application, limits=limits)
        assert not expired and applied.axes == pytest.approx([.5, .25, .5, 0.])


# 功能：
#   验证动作变化代价采用固定 NED 轴并正确环绕航向，首次回执只建立基准。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_transport_change_cost_uses_actual_fixed_frame_command_and_wraps_heading():
    _, _, application, limits = teacher_evidence()
    assert transport_command_change_squared(None, application, limits=limits) == 0
    first = application.model_copy(update={"velocity_ned_mps": (0., 0., 0.),
                                           "yaw_heading_deg": 179., "yaw_rate_application": None})
    second = application.model_copy(update={"velocity_ned_mps": (.8, 0., .375),
                                            "yaw_heading_deg": -179., "yaw_rate_application": None})
    assert transport_command_change_squared(first, second, limits=limits) == (
        pytest.approx(1.25 + (2./180)**2))


# 功能：
#   验证提议、错误运输模式、过期输入及错误绑定不能成为教师标签。
# 输入：
#   update：对回执施加的错误字段变更。
#   issue：应出现的具体拒绝代码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "update,issue",
    [
        ({"transport": "position-velocity-ned", "position_ned_m": (1., 2., 3.)},
         "POSITION_CONTROL"),
        ({"yaw_rate_application": None}, "YAW_RATE_MISSING"),
        ({"accepted_at_unix_ms": 999}, "TIME_ALIGNMENT"),
        ({"accepted_at_unix_ms": 1300}, "TIME_ALIGNMENT"),
        ({"command_sha256": "d" * 64}, "COMMAND_MISMATCH"),
        ({"velocity_ned_mps": (0.0, 0.0, 0.0)}, "VELOCITY_DIFFERS"),
        ({"model_authorized": True}, "COMMAND_MISMATCH"),
    ],
)
def test_proposals_wrong_transports_and_stale_evidence_are_not_labels(update, issue):
    snapshot, command, application, limits = teacher_evidence()
    altered = type(application).model_validate({**application.model_dump(), **update})
    with pytest.raises(ValueError, match=issue):
        executed_pilot_control(snapshot, command, altered, limits=limits)


# 功能：
#   验证越出部署包络的实际控制被拒绝，而非裁剪为并未真正执行的标签。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_out_of_envelope_control_is_not_silently_clipped():
    snapshot, command, application, _ = teacher_evidence()
    task = snapshot["strategic_context"]["task"]
    task["normalized_pilot_control_limits"]["horizontal_speed_mps"] = 0.1
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    with pytest.raises(ValueError, match="less than or equal"):
        executed_pilot_control(
            snapshot, command, application, limits=PilotControlLimits(0.1, 0.75, 20)
        )


# 功能：
#   验证修改目标或归一化尺度不能重新解释旧教师样本。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_changing_goal_or_normalization_cannot_relabel_old_teacher_input():
    snapshot, command, application, limits = teacher_evidence()
    snapshot["goal_position_m"]["x"] = 0.1
    with pytest.raises(ValueError, match="SNAPSHOT_HASH"):
        executed_pilot_control(snapshot, command, application, limits=limits)
    snapshot, command, application, _ = teacher_evidence()
    with pytest.raises(ValueError, match="DEPLOYMENT_LIMITS"):
        executed_pilot_control(
            snapshot, command, application, limits=PilotControlLimits(0.5, 0.75, 20)
        )
