"""Transport/evidence contract tests, not physical outcome receipts."""

import copy

import pytest
from control_fixtures import complete_feature_snapshot
from test_runtime_evidence_inventory import receipt

from dronedream_agent_core.contracts import (
    BodyFrameControlIntent,
    PredictiveSafetyDecision,
    QuaternionWxyz,
    RuntimeLocalSafetyCommand,
    Vector3,
)
from dronedream_agent_core.control_execution_evidence import (
    ControlApplicationRecord,
    control_application_record,
    validate_application_binding,
    verify_control_applications,
)
from dronedream_agent_core.hashing import sha256_json


# 功能：
#   构造彼此绑定的输入快照、模型意图、批准命令与实际传输回执，用于离线验收测试。
# 输入：
#   无。
# 输出：
#   record：实际传输回执字典。
#   inputs：快照、命令、写入清单及预期记录数。
def evidence():
    snapshot = {"realtime_feature_snapshot": complete_feature_snapshot().model_dump(mode="json")}
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    intent = BodyFrameControlIntent(
        source_expert="local-navigation-policy", control_origin="continuous-model-output",
        model_call_id="model-" + "a" * 24,
        navigation_snapshot_sha256=snapshot["snapshot_sha256"], task_reference_sha256="b" * 64,
        generated_at_unix_ms=1000, valid_until_unix_ms=1200, forward_velocity_mps=.5,
        right_velocity_mps=0, up_velocity_mps=.1, yaw_rate_dps=5,
        maximum_acceleration_mps2=1, maximum_jerk_mps3=2,
    )
    command = RuntimeLocalSafetyCommand(
        observation_sha256="c" * 64, observation_sequence=1, generated_at_unix_ms=1050,
        valid_until_unix_ms=1200, source="onboard",
        evaluated_body_orientation_world_from_body=QuaternionWxyz(w=1, x=0, y=0, z=0),
        evaluated_target_position_m=Vector3(x=1,y=2,z=3),
        navigation_goal_id="office", navigation_control_authority="model-required",
        model_navigation_authorized=True, model_call_id=intent.model_call_id,
        model_path_sha256=intent.task_reference_sha256,
        model_navigation_snapshot_sha256=intent.navigation_snapshot_sha256,
        requested_control_intent=intent, command_position_m=Vector3(x=1,y=2,z=3),
        decision=PredictiveSafetyDecision(
            action="continue", selected_velocity_mps=Vector3(x=.1,y=.2,z=.05),
            selected_yaw_rate_dps=5, control_source="local-model-body-control",
            predicted_path_m=[Vector3(x=1,y=2,z=3)], minimum_predicted_clearance_m=1,
            time_to_minimum_clearance_seconds=.1, evaluated_candidate_count=1,
        ),
    )
    record = control_application_record(command, sequence=1, accepted_at_unix_ms=1100,
        transport="velocity-ned", velocity_ned_mps=(.2,.1,-.05),
        yaw_heading_deg=20, yaw_rate_application={
            "previous_heading_deg": 19.75, "clockwise_rate_dps": 5,
            "integration_seconds": .05,
        }).model_dump(mode="json")
    counts = {"control-applications.jsonl": 1, "local-safety-executor-history.jsonl": 2}
    inputs = {"navigation_snapshots": [{"snapshot": snapshot}],
        "command_records": [{"command": command.model_dump(mode="json")}], "expected_count": 1,
        "writer_summary": receipt(counts), "writer_artifact_counts": counts,
        "application_artifact_path": "control-applications.jsonl"}
    return record, inputs


# 功能：
#   验证合法实际传输回执能够绑定原始输入年龄、命令与授权期限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_actual_model_command_binds_input_deadline_and_transport():
    record, inputs = evidence()
    result = verify_control_applications([record], **inputs)
    assert result["accepted"]
    assert result["maximum_model_input_age_ms"] == 100


@pytest.mark.parametrize("fault", [None, "velocity", "position", "late"])
def test_hybrid_receipt_is_verified_without_counting_as_model(fault):
    """功能：核对衔接实际发送；输入：合法/篡改回执；输出：独立计数及对应拒绝。"""
    from test_bounded_hybrid_control import bridge_command
    record, inputs = evidence()
    command = bridge_command()
    record = control_application_record(command, sequence=1, accepted_at_unix_ms=1100,
        transport="velocity-ned", velocity_ned_mps=(.1, .1, 0.), yaw_heading_deg=0.).model_dump(mode="json")
    inputs["command_records"] = [{"command": command.model_dump(mode="json")}]
    if fault == "velocity":
        record["velocity_ned_mps"] = [10., 0., 0.]
    elif fault == "position":
        record.update(transport="position-velocity-ned", position_ned_m=[2., 0., 0.])
    elif fault == "late":
        record["accepted_at_unix_ms"] = 2501
    result = verify_control_applications([record], **inputs)
    assert result["model_motion_record_count"] == 0
    assert result["bounded_hybrid_motion_record_count"] == 1
    expected = {"CONTROL_APPLICATION_HAS_NO_MODEL_MOTION"}
    if fault == "velocity":
        expected.add("CONTROL_APPLICATION_VELOCITY_DIFFERS_FROM_APPROVED_COMMAND")
    elif fault == "position":
        expected.add("CONTROL_APPLICATION_HYBRID_TRANSPORT_INVALID")
    elif fault == "late":
        expected.update(("CONTROL_APPLICATION_TRANSPORT_DEADLINE_VIOLATION",
                         "CONTROL_APPLICATION_HYBRID_AUTHORITY_EXPIRED"))
    assert set(result["issue_codes"]) == expected


# 功能：
#   明确未发布的过期计算不污染有效回执；它也不能补出缺失的已执行命令。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_explicit_absent_publication_is_not_a_corrupt_or_executed_command():
    record, inputs = evidence()
    inputs["command_records"].append({"recorded_at_unix_ms": 1010, "command": None})
    assert verify_control_applications([record], **inputs)["accepted"]
    inputs["command_records"] = [{"recorded_at_unix_ms": 1010, "command": None}]
    result = verify_control_applications([record], **inputs)
    assert not result["accepted"]
    assert "CONTROL_APPLICATION_COMMAND_MISSING" in result["issue_codes"]


# 功能：
#   丢失 command 字段、畸形命令或非对象行仍按损坏证据拒绝，不能当成显式未发布。
# 输入：
#   row：损坏的命令信封。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("row", [{}, {"command": []}, {"command": False}, None])
def test_missing_or_malformed_command_is_not_an_absent_publication(row):
    record, inputs = evidence()
    inputs["command_records"].append(row)
    result = verify_control_applications([record], **inputs)
    assert "CONTROL_APPLICATION_COMMAND_EVIDENCE_INVALID" in result["issue_codes"]


# 功能：
#   验证控制回执流不能借用其他证据流的记录数掩盖丢失或未落盘。
# 输入：
#   fault：注入清单的损坏类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["swapped-counts", "unknown-path", "missing-stream", "truncated",
                                    "duplicate", "undrained", "untyped-count"])
def test_control_receipt_counts_cannot_borrow_the_other_writer_stream(fault):
    record, inputs = evidence()
    # One actual receipt plus two executor history rows is a complete FIFO,
    # not a count mismatch. Same totals alone must not rescue corrupt streams.
    assert verify_control_applications([record], **inputs)["accepted"]
    report = inputs["writer_summary"]
    if fault == "swapped-counts":
        report["artifacts"][0].update(submitted_count=2, completed_count=2)
        report["artifacts"][1].update(submitted_count=1, completed_count=1)
    elif fault == "unknown-path":
        report["artifacts"][1]["path"] = "elsewhere/history.jsonl"
    elif fault == "missing-stream":
        del inputs["writer_artifact_counts"]["local-safety-executor-history.jsonl"]
    elif fault == "truncated":
        inputs["writer_artifact_counts"]["local-safety-executor-history.jsonl"] = -1
    elif fault == "duplicate":
        report["artifacts"].append(dict(report["artifacts"][0]))
    elif fault == "undrained":
        report["pending_count"] = 1
    elif fault == "untyped-count":
        inputs["writer_artifact_counts"]["control-applications.jsonl"] = True
    result = verify_control_applications([record], **inputs)
    assert not result["accepted"]
    assert "CONTROL_APPLICATION_EVIDENCE_NOT_DRAINED" in result["issue_codes"]


# 功能：
#   验证过期、错序、指令被替换或偏航证据缺失时，验收给出对应拒绝原因。
# 输入：
#   field：被替换字段。
#   value：错误替换值。
#   issue：应出现的拒绝代码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value,issue", [
    ("accepted_at_unix_ms", 1300, "CONTROL_APPLICATION_INPUT_DEADLINE_VIOLATION"),
    ("transport", "position-velocity-ned", "CONTROL_APPLICATION_NOT_DIRECT_MODEL_CONTROL"),
    ("velocity_ned_mps", [.9, .1, -.05],
     "CONTROL_APPLICATION_VELOCITY_DIFFERS_FROM_APPROVED_COMMAND"),
    ("sequence", 2, "CONTROL_APPLICATION_ORDER_INVALID"),
    ("command_sha256", "e" * 64, "CONTROL_APPLICATION_COMMAND_MISSING"),
    ("observation_sha256", "f" * 64, "CONTROL_APPLICATION_COMMAND_BINDING_MISMATCH"),
    ("yaw_rate_application", None, "CONTROL_APPLICATION_YAW_AUTHORITY_MISMATCH"),
    ("yaw_heading_deg", 25, "CONTROL_APPLICATION_EVIDENCE_INVALID"),
])
def test_bad_control_execution_evidence_fails_closed(field, value, issue):
    record, inputs = evidence()
    record[field] = value
    if field == "transport":
        # 提供合法位置，使测试到达“不是直接速度控制”的门槛，而非先失败于缺失字段。
        record["position_ned_m"] = [2., 1., -3.]
    result = verify_control_applications([record], **inputs)
    assert not result["accepted"] and issue in result["issue_codes"]


# 功能：
#   验证改写快照采集时刻而未保持内容摘要一致时，不能刷新模型输入期限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_sensor_snapshot_hash_cannot_be_rewritten_to_renew_input():
    record, inputs = evidence()
    altered = copy.deepcopy(inputs)
    features = altered["navigation_snapshots"][0]["snapshot"]["realtime_feature_snapshot"]
    features["captured_at_unix_ms"] = 1100
    result = verify_control_applications([record], **altered)
    assert "CONTROL_APPLICATION_MODEL_INPUT_MISSING" in result["issue_codes"]


# 功能：
#   验证没有执行回执或尚未完整落盘时，不能宣称模型完成控制。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_missing_execution_or_undrained_records_do_not_count_as_control():
    record, inputs = evidence()
    inputs["writer_summary"]["complete"] = False
    result = verify_control_applications([record], **inputs)
    assert "CONTROL_APPLICATION_EVIDENCE_NOT_DRAINED" in result["issue_codes"]
    result = verify_control_applications([], **inputs)
    assert "CONTROL_APPLICATION_HAS_NO_MODEL_MOTION" in result["issue_codes"]


# 功能：
#   验证连续控制命令拒绝仿真真值拟合偏移，避免旧对齐方式混入当前控制证据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_continuous_command_rejects_legacy_runtime_truth_alignment():
    _, inputs = evidence()
    command = inputs["command_records"][0]["command"]
    command["estimator_to_world_position_offset_m"] = {"x": .1, "y": 0, "z": 0}
    with pytest.raises(ValueError, match="truth-fitted"):
        RuntimeLocalSafetyCommand.model_validate(command)


# 功能：
#   验证确定性安全替代单独计数，并同样检查速度、偏航及输入有效期。
# 输入：
#   fault：安全替代回执的损坏类型，None 表示合法。
#   issue：对应的拒绝代码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault,issue", [
    (None, None),
    ("missing-yaw", "CONTROL_APPLICATION_YAW_AUTHORITY_MISMATCH"),
    ("turned", "CONTROL_APPLICATION_YAW_AUTHORITY_MISMATCH"),
    ("late", "CONTROL_APPLICATION_INPUT_DEADLINE_VIOLATION"),
    ("velocity", "CONTROL_APPLICATION_VELOCITY_DIFFERS_FROM_APPROVED_COMMAND"),
    ("missing-input", "CONTROL_APPLICATION_MODEL_INPUT_MISSING"),
])
def test_safety_velocity_is_verified_without_counting_as_model_control(fault, issue):
    record, inputs = evidence()
    payload = copy.deepcopy(inputs["command_records"][0]["command"])
    payload["decision"].update(action="slow", control_source="deterministic-safety-override",
                               selected_yaw_rate_dps=1 if fault == "turned" else 0)
    command = RuntimeLocalSafetyCommand.model_validate(payload)
    yaw = {"previous_heading_deg": 114., "clockwise_rate_dps": 0., "integration_seconds": .05}
    override = control_application_record(command, sequence=2, accepted_at_unix_ms=1101,
        transport="velocity-ned", velocity_ned_mps=(.2, .1, -.05), yaw_heading_deg=114.,
        yaw_rate_application=yaw).model_dump(mode="json")
    inputs["command_records"].append({"command": command.model_dump(mode="json")})
    inputs["writer_artifact_counts"]["control-applications.jsonl"] = 2
    inputs.update(expected_count=2, writer_summary=receipt(inputs["writer_artifact_counts"]))
    if fault == "missing-yaw":
        override["yaw_rate_application"] = None
    elif fault == "late":
        override["accepted_at_unix_ms"] = 1300
    elif fault == "velocity":
        override["velocity_ned_mps"] = [.9, .1, -.05]
    elif fault == "missing-input":
        inputs["navigation_snapshots"] = []
    result = verify_control_applications([record, override], **inputs)
    assert result["model_motion_record_count"] == 1
    assert result["safety_velocity_record_count"] == 1
    assert result["accepted"] == (fault is None)
    if issue:
        assert issue in result["issue_codes"]


# 功能：
#   验证多余的损坏证据不被静默忽略，也不会在散列或迭代时令验收崩溃。
# 输入：
#   stream：命令或导航快照证据流名称。
#   envelope：要插入的损坏信封。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("stream,envelope", [
    ("command_records", {}), ("command_records", {"command": []}),
    ("navigation_snapshots", {}), ("navigation_snapshots", {"snapshot": []}),
    ("navigation_snapshots", {"snapshot": {"snapshot_sha256": "a" * 64, "x": float("nan")}}),
])
def test_corrupt_extra_envelopes_are_not_silently_dropped(stream, envelope):
    record, inputs = evidence()
    inputs[stream].append(envelope)
    result = verify_control_applications([record], **inputs)
    assert not result["accepted"]


# 功能：
#   验证期望条数不接受布尔或浮点值，即使它们与整数条数比较相等。
# 输入：
#   count：非法期望条数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("count", [True, 1., "1", None, -1])
def test_expected_count_requires_an_integer(count):
    record, inputs = evidence()
    inputs["expected_count"] = count
    assert not verify_control_applications([record], **inputs)["accepted"]


# 功能：
#   验证顶层证据容器或路径损坏时返回拒绝报告，而不是泄漏 Python 类型异常。
# 输入：
#   field：被替换的证据参数。
#   value：非法参数值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [
    ("records", None), ("records", {}), ("navigation_snapshots", None),
    ("command_records", None), ("application_artifact_path", []),
])
def test_invalid_top_level_evidence_is_rejected(field, value):
    record, inputs = evidence()
    records = [record]
    if field == "records":
        records = value
    else:
        inputs[field] = value
    assert not verify_control_applications(records, **inputs)["accepted"]


# 功能：
#   验证从文件直接回读位置控制回执时，也必须具有实际位置设定值。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_position_receipt_requires_setpoint_on_readback():
    record, _ = evidence()
    record["transport"] = "position-velocity-ned"
    with pytest.raises(ValueError, match="ACTUAL_SETPOINT"):
        ControlApplicationRecord.model_validate(record)


# 功能：
#   验证偏航回读采用与控制器相同的先取模再积分算法，避免巨大累计角度吞掉小转角。
# 输入：
#   previous：控制器保留的累计航向，单位度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("previous", [1e20, -1e20, 1e308, -1e308])
def test_yaw_receipt_preserves_small_turns_at_large_accumulated_headings(previous):
    record, _ = evidence()
    record["yaw_rate_application"]["previous_heading_deg"] = previous
    record["yaw_heading_deg"] = (previous % 360. + .25 + 180.) % 360. - 180.
    accepted = ControlApplicationRecord.model_validate(record)
    assert accepted.yaw_heading_deg == record["yaw_heading_deg"]
    record["yaw_heading_deg"] += 1
    with pytest.raises(ValueError, match="YAW_CONVERSION_MISMATCH"):
        ControlApplicationRecord.model_validate(record)


# 功能：
#   验证回执中的物理量及授权标志不允许由字符串或布尔值隐式转换。
# 输入：
#   field：被替换的回执字段。
#   value：不符合字段语义的值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [
    ("model_authorized", "true"), ("model_authorized", 1),
    ("velocity_ned_mps", [True, .1, -.05]), ("velocity_ned_mps", [".2", .1, -.05]),
    ("yaw_heading_deg", "20"),
    ("yaw_rate_application", {"previous_heading_deg": "19.75", "clockwise_rate_dps": 5,
                              "integration_seconds": .05}),
])
def test_receipt_physical_values_are_not_coerced(field, value):
    record, _ = evidence()
    record[field] = value
    with pytest.raises(ValueError):
        ControlApplicationRecord.model_validate(record)


# 功能：
#   验证已捕获的实际执行回执拥有自己的意图副本，不随后续命令对象修改而漂移。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_application_captures_an_independent_intent():
    _, inputs = evidence()
    command = RuntimeLocalSafetyCommand.model_validate(inputs["command_records"][0]["command"])
    application = control_application_record(
        command, sequence=1, accepted_at_unix_ms=1100, transport="velocity-ned",
        velocity_ned_mps=(.2, .1, -.05), yaw_heading_deg=20,
    )
    before = application.model_dump(mode="json")
    command.requested_control_intent.forward_velocity_mps = .9
    assert application.model_dump(mode="json") == before


# 功能：
#   验证训练侧绑定入口重新检查被原地修改的回执，不能只比较命令摘要。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_binding_revalidates_mutated_receipt():
    record, inputs = evidence()
    command = RuntimeLocalSafetyCommand.model_validate(inputs["command_records"][0]["command"])
    application = ControlApplicationRecord.model_validate(record)
    validate_application_binding(command, application)
    application.yaw_rate_application.clockwise_rate_dps = 10
    with pytest.raises(ValueError, match="YAW_CONVERSION_MISMATCH"):
        validate_application_binding(command, application)
