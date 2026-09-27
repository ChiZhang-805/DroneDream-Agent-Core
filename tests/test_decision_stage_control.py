"""Mixed implementation is distinct from pure model evidence; synthetic fixtures only."""

import pytest
from test_decision_label_evidence import native_fixture

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand
from dronedream_agent_core.control_execution_evidence import control_application_record
from dronedream_agent_core.decision_stage_control import verify_stage_control


# 功能：构造真实模式名称的短制动夹具，不声称仿真飞行；输入：无；输出：合成记录。
def stage_fixture():
    row, document = native_fixture()
    document["behavior_application"]["controller_mode"] = "model-with-bounded-hybrid"
    raw = document["commands"][4]
    raw.update(
        model_navigation_authorized=False,
        model_call_id=None,
        model_path_sha256=None,
        model_navigation_snapshot_sha256=None,
        requested_control_intent=None,
    )
    raw["decision"].update(
        action="hold",
        control_source="deterministic-brake",
        selected_velocity_mps={"x": 0.0, "y": 0.0, "z": 0.0},
    )
    command = RuntimeLocalSafetyCommand.model_validate(raw)
    document["control_applications"][4] = control_application_record(
        command,
        sequence=5,
        accepted_at_unix_ms=10800,
        transport="velocity-ned",
        velocity_ned_mps=(0.0, 0.0, 0.0),
        yaw_heading_deg=0.0,
    ).model_dump(mode="json")
    return row, document


# 功能：短制动保持来源区别且不自动授予正标签；输入：合成回执；输出：分别计数。
def test_bounded_measured_brake_is_not_relabelled_as_model():
    row, document = stage_fixture()
    result = verify_stage_control(row, document)
    assert result["model"] == 10 and result["measured_brake"] == 1
    assert result["formal_training_additions"] == 0
    assert not document["control_applications"][4]["model_authorized"]


# 功能：缺传感器时的独立制动必须实际落盘并放回原传输顺序，不能补写为模型动作。
# 输入：命令间的合成真实传输回执及顺序故障；输出：正确分类或明确拒绝。
@pytest.mark.parametrize("fault", [None, "anchor", "time", "sequence", "gap"])
def test_transport_only_brakes_have_independent_order(fault):
    from dronedream_agent_core.executor_brake_evidence import ExecutorBrakeApplication

    row, document = stage_fixture()
    brakes = [
        ExecutorBrakeApplication(
            sequence=i + 4,
            after_command_application_sequence=3,
            accepted_at_unix_ms=10450 + i * 50,
            reason="command-stale",
            position_ned_m=(0.0, 0.1, -1.2),
            yaw_heading_deg=0.0,
        ).model_dump(mode="json")
        for i in range(2)
    ]
    document["executor_brake_applications"] = brakes
    if fault == "anchor":
        brakes[1]["after_command_application_sequence"] = 2
    elif fault == "time":
        brakes[0]["accepted_at_unix_ms"] = 10650
    elif fault == "sequence":
        brakes[1]["sequence"] += 1
    elif fault == "gap":
        # 完整命令序号仍然必须保留，不能借制动记录把删掉的模型动作补上。
        document["control_applications"].pop(3)
    if fault:
        with pytest.raises(ValueError):
            verify_stage_control(row, document)
    else:
        result = verify_stage_control(row, document)
        assert result["model"] == 10 and result["measured_brake"] == 3
        assert all(not b["model_authorized"] for b in brakes)


# 功能：独立桥接按路线租期审计，不强迫它冒充某次模型输出；输入：合成桥接；输出：分类及拒绝变体。
@pytest.mark.parametrize("fault", [None, "route", "origin", "displacement"])
def test_route_bridge_does_not_require_fabricated_model_pointer(fault):
    from dronedream_agent_core.contracts import RuntimeLocalSafetyObservation

    row, document = stage_fixture()
    # 复用本文件的合成安全命令，不引入执行器/仿真测试模块到云端训练包。
    raw = document["commands"][4]
    raw["navigation_control_authority"] = "bounded-hybrid"
    raw["observation_budget"] = dict(
        source_observed_at_unix_ms=10790,
        control_deadline_unix_ms=11000,
        disposition="control-eligible",
        reason="fresh-synthetic-observation",
        clearance_margin_m=1.0,
        uncertainty_margin_m=0.1,
        downstream_reserve_ms=40,
    )
    raw["hybrid_lease"] = dict(
        navigation_goal_id=row["state"]["goal_id"],
        route_sha256=row["state"]["route_sha256"],
        episode=1,
        started_at_unix_ms=10800,
        expires_at_unix_ms=12300,
        maximum_speed_mps=0.25,
        maximum_distance_m=0.35,
    )
    raw["decision"].update(
        action="continue",
        control_source="route-target",
        selected_velocity_mps=dict(x=0.1, y=0.1, z=0.0),
    )
    raw["navigation_goal_id"] = row["state"]["goal_id"]
    raw["hybrid_lease"].update(
        navigation_goal_id=row["state"]["goal_id"], route_sha256=row["state"]["route_sha256"]
    )
    raw.update(model_call_id=None, model_path_sha256=None)
    if fault == "route":
        raw["hybrid_lease"]["route_sha256"] = "f" * 64
    elif fault == "origin":
        raw["hybrid_lease"].update(started_at_unix_ms=10400, expires_at_unix_ms=11900)
    command = RuntimeLocalSafetyCommand.model_validate(raw)
    document["commands"][4] = command.model_dump(mode="json")
    document["control_applications"][4] = control_application_record(
        command,
        sequence=5,
        accepted_at_unix_ms=10800,
        transport="velocity-ned",
        velocity_ned_mps=(0.1, 0.1, 0.0),
        yaw_heading_deg=0.0,
    ).model_dump(mode="json")
    before = document["observations"][4]
    middle = RuntimeLocalSafetyObservation.model_validate(
        dict(
            before,
            sequence=100,
            observed_at_unix_ms=10900,
            current_position_m=dict(x=0.7 if fault == "displacement" else 0.225, y=0.0, z=1.5),
        )
    ).model_dump(mode="json")
    document["hybrid_displacement_observations"] = (
        document["observations"][:5] + [middle] + document["observations"][5:]
    )
    if fault is not None:
        with pytest.raises(ValueError):
            verify_stage_control(row, document)
    else:
        result = verify_stage_control(row, document)
        assert result["bounded_bridge"] == 1 and result["model"] == 10
        assert document["commands"][4]["model_call_id"] is None


# 功能：删记录、强行改名或缺模型交还均拒绝；输入：错误变体；输出：不得作为合法阶段来源。
@pytest.mark.parametrize("fault", ["gap", "first", "last", "moving_brake", "action", "mode"])
def test_invalid_handoff_is_not_a_stage_implementation(fault):
    row, document = stage_fixture()
    if fault == "gap":
        document["control_applications"].pop(4)
    elif fault in {"first", "last"}:
        document["control_applications"] = (
            document["control_applications"][4:]
            if fault == "first"
            else document["control_applications"][:5]
        )
    elif fault == "moving_brake":
        document["control_applications"][4]["velocity_ned_mps"] = [0.1, 0.0, 0.0]
    elif fault == "action":
        document["behavior_application"]["action"] = "wait"
    else:
        document["behavior_application"].pop("controller_mode")
    with pytest.raises(ValueError):
        verify_stage_control(row, document)
