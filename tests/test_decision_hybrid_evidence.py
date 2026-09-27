"""Explicit synthetic distance witnesses, never formal decision demonstrations."""

import pytest
from test_bounded_hybrid_control import bridge_command

from dronedream_agent_core.contracts import RuntimeLocalSafetyObservation, Vector3
from dronedream_agent_core.control_execution_evidence import control_application_record
from dronedream_agent_core.decision_hybrid_evidence import audit_hybrid_displacement


# 功能：构造短租期和独立观测夹具；输入：无；输出：非正式测试记录。
def fixture():
    command = bridge_command()
    applications = [
        control_application_record(
            command,
            sequence=i + 1,
            accepted_at_unix_ms=1000 + i * 80,
            transport="velocity-ned",
            velocity_ned_mps=(0.1, 0.1, 0.0),
            yaw_heading_deg=0.0,
        ).model_dump(mode="json")
        for i in range(3)
    ]
    observations = [
        RuntimeLocalSafetyObservation(
            sequence=i + 1,
            observed_at_unix_ms=1000 + i * 80,
            source="simulation-ground-truth",
            stream_healthy=True,
            stream_age_seconds=0.0,
            localization_covariance_m2=0.0,
            current_position_m=Vector3(x=i * 0.01, y=0.0, z=1.0),
            current_velocity_mps=Vector3(x=0.1, y=0.0, z=0.0),
            target_position_m=Vector3(x=1.0, y=0.0, z=1.0),
        ).model_dump(mode="json")
        for i in range(3)
    ]
    return [{"command": command.model_dump(mode="json")}], applications, observations


# 功能：真实路径采样和指令速度分开计量；输入：合成小位移；输出：仍不授予正式标签。
def test_distance_is_not_integrated_command_speed():
    result = audit_hybrid_displacement(*fixture())
    assert result["episodes"][0]["sampled_path_length_m"] == pytest.approx(0.02)
    assert not result["qualification_granted"] and result["formal_training_additions"] == 0


# 功能：严格阶段验收不能在最后一次桥接发送处截断运动；输入：没有模型接回的片段；输出：拒绝。
def test_stage_audit_requires_actual_model_handback():
    with pytest.raises(ValueError, match="MODEL_HANDBACK_MISSING_OR_LATE"):
        audit_hybrid_displacement(*fixture(), require_model_handback=True)


# 功能：桥接结束至真正接回之间的移动同样消耗距离；输入：合成可回放记录；输出：完整区间核验。
@pytest.mark.parametrize("extra_motion", [False, True])
def test_strict_distance_includes_motion_before_handback(extra_motion):
    from test_decision_label_evidence import native_fixture
    from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand

    command = bridge_command(now=10000)
    commands = [{"command": command.model_dump(mode="json")}]
    actuals = [control_application_record(command, sequence=1, accepted_at_unix_ms=10000,
        transport="velocity-ned", velocity_ned_mps=(0.1, 0.1, 0.0),
        yaw_heading_deg=0.0).model_dump(mode="json")]
    _, model = native_fixture()
    raw = model["commands"][1]
    raw["navigation_goal_id"] = "office"
    handback = RuntimeLocalSafetyCommand.model_validate(raw)
    commands.append({"command": handback.model_dump(mode="json")})
    actuals.append(control_application_record(handback, sequence=2, accepted_at_unix_ms=10200,
        transport="velocity-ned", velocity_ned_mps=(0.0, 0.25, 0.0),
        yaw_heading_deg=0.0).model_dump(mode="json"))
    observations = model["observations"][:2]
    middle = dict(observations[0], observed_at_unix_ms=10100,
                  current_position_m=dict(x=0.4 if extra_motion else 0.025, y=0.0, z=1.5))
    observations.insert(1, middle)
    if extra_motion:
        with pytest.raises(ValueError, match="DISPLACEMENT_BUDGET_EXCEEDED"):
            audit_hybrid_displacement(commands, actuals, observations, require_model_handback=True)
    else:
        result = audit_hybrid_displacement(commands, actuals, observations, require_model_handback=True)
        assert result["episodes"][0]["audited_until_ms"] == 10200
        assert result["episodes"][0]["sampled_path_length_m"] == pytest.approx(0.05)


# 功能：来回运动不能以净位移为零逃避额度；输入：先前进再返回；输出：超额拒绝。
def test_round_trip_motion_consumes_distance_budget():
    commands, actuals, observations = fixture()
    observations[1]["current_position_m"]["x"] = 0.2
    observations[2]["current_position_m"]["x"] = 0.0
    with pytest.raises(ValueError, match="DISPLACEMENT_BUDGET_EXCEEDED"):
        audit_hybrid_displacement(commands, actuals, observations)


# 功能：缺帧、估计值代替真值、错速度和时钟倒序均失败；输入：变异；输出：明确拒绝。
@pytest.mark.parametrize("fault", ["missing", "source", "velocity", "time"])
def test_bad_evidence_is_rejected(fault):
    commands, actuals, observations = fixture()
    if fault == "missing":
        observations.pop(1)
    elif fault == "source":
        observations[0]["source"] = "onboard"
    elif fault == "velocity":
        actuals[0]["velocity_ned_mps"] = [0.0, 0.1, 0.0]
    else:
        actuals.reverse()
    with pytest.raises(ValueError):
        audit_hybrid_displacement(commands, actuals, observations)
