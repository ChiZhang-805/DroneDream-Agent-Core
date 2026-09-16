"""飞行数据合同的离线反例，不以结构合法代替传感器真实性或飞行资格。"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from dronedream_agent_core.contracts import (
    FlightPlan,
    GraphRoute,
    LocalPlannerRequest,
    PlanSegment,
    PluginInvocation,
    PredictiveSafetyDecision,
    QuaternionWxyz,
    RoutePoint,
    RuntimeActionAdapterCatalog,
    RuntimeActionAdapterDefinition,
    RuntimeLocalSafetyCommand,
    RuntimeLocalSafetyObservation,
    RuntimeOperatorTakeoverGrant,
    Vector3,
    VehicleAsset,
)


# 功能：
#   构造带明确坐标的测试点，避免各用例重复填写无关轴。
# 输入：
#   x：测试点的东向坐标，单位米。
# 输出：
#   point：离线测试三维点。
def _point(x=0.0):
    point = Vector3(x=x, y=0.0, z=1.0)
    return point


# 功能：
#   验证有限但极大的四元数被归一化校验拒绝，而非泄漏 OverflowError。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_extreme_quaternion_is_a_validation_error():
    with pytest.raises(ValidationError, match="normalized"):
        QuaternionWxyz(w=1e308, x=0, y=0, z=0)


# 功能：
#   验证插件参数不能借重复键或非有限数制造解析歧义。
# 输入：
#   arguments：错误 JSON 参数文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("arguments", ['{"speed":1,"speed":10}', '{"x":NaN}', '{"x":1e999}'])
def test_plugin_arguments_reject_ambiguous_json(arguments):
    invocation = PluginInvocation(tool_id="test.tool", arguments_json=arguments, purpose="test")
    with pytest.raises(ValueError):
        invocation.parsed_arguments()


# 功能：
#   验证插件参数按 UTF-8 字节而非仅字符数控制实际传输量。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_plugin_arguments_have_a_byte_budget():
    invocation = PluginInvocation(
        tool_id="test.tool", arguments_json='{"text":"' + "测" * 30_000 + '"}', purpose="test"
    )
    with pytest.raises(ValueError, match="SIZE_LIMIT"):
        invocation.parsed_arguments()


# 功能：
#   验证合法参数仍能保留标准 JSON 的数值、布尔、数组及空值。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_plugin_arguments_preserve_valid_values():
    invocation = PluginInvocation(
        tool_id="test.tool", arguments_json='{"speed":0.2,"flags":[true,null]}', purpose="test"
    )
    assert invocation.parsed_arguments() == {"speed": 0.2, "flags": [True, None]}


# 功能：
#   验证最大起飞总重不能低于空机重量。
# 输入：
#   masses：空机、最大起飞及最大取件载荷重量，单位公斤。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("masses", [(2, 1, 0)])
def test_vehicle_mass_limits_are_consistent(masses):
    with pytest.raises(ValidationError, match="mass|payload"):
        VehicleAsset(
            asset_id="vehicle",
            name="test",
            dry_mass_kg=masses[0],
            max_takeoff_mass_kg=masses[1],
            max_pickup_payload_kg=masses[2],
            body_radius_m=0.2,
            body_height_m=0.3,
            max_speed_mps=2,
            max_acceleration_mps2=2,
            qualified_range_m=500,
            reserve_battery_percent=20,
            sensors=["camera"],
        )


# 功能：
#   验证路线标识、点序列和边数量必须对应，不能拼成自相矛盾的导航输入。
# 输入：
#   change：替换的路线字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "change",
    [
        {"positions_m": [_point()]},
        {"edge_ids": []},
        {"start_node": "unknown"},
        {"goal_node": "unknown"},
    ],
)
def test_route_arrays_and_endpoints_agree(change):
    values = dict(
        start_node="a",
        goal_node="b",
        node_ids=["a", "b"],
        edge_ids=["ab"],
        positions_m=[_point(), _point(1)],
        route_length_m=1,
        all_edges_flight_verified=False,
    )
    values.update(change)
    with pytest.raises(ValidationError):
        GraphRoute(**values)


# 功能：
#   验证返程仍可重复访问节点，不把合乎任务要求的往返当成非法路线。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_route_allows_return_to_the_start():
    route = GraphRoute(
        start_node="a",
        goal_node="a",
        node_ids=["a", "b", "a"],
        edge_ids=["ab", "ab"],
        positions_m=[_point(), _point(1), _point()],
        route_length_m=2,
        all_edges_flight_verified=False,
    )
    assert route.node_ids == ["a", "b", "a"]


# 功能：
#   构造几何连续、可用于核对跨段接口的短航段。
# 输入：
#   identifier：当前航段编号。
#   start：航段起始节点名称。
#   end：航段终点名称。
#   positions：起终点的三维坐标。
# 输出：
#   segment：离线航段合同。
def _segment(identifier, start, end, positions):
    segment = PlanSegment(
        segment_id=identifier,
        task_id="move",
        from_node=start,
        to_node=end,
        path=[
            RoutePoint(node_id=start, position_m=positions[0]),
            RoutePoint(node_id=end, position_m=positions[1]),
        ],
        speed_limit_mps=1,
        minimum_clearance_m=0.2,
        success_evidence=["position"],
    )
    return segment


# 功能：
#   验证同名接点的位置跳变和重复航段身份不能进入飞行计划。
# 输入：
#   defect：待构造的航段身份或几何错误。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("defect", ["position", "identity"])
def test_flight_plan_binds_geometry_and_segment_identity(defect):
    first = _segment("segment-001", "a", "b", [_point(), _point(1)])
    second = _segment(
        "segment-001" if defect == "identity" else "segment-002",
        "b",
        "c",
        [_point(9 if defect == "position" else 1), _point(2)],
    )
    with pytest.raises(ValidationError):
        FlightPlan(
            revision=1, contract_id="test", segments=[first, second], semantic_plan_sha256="a" * 64
        )


# 功能：
#   验证长时间未更新的观测可保留真实年龄供安全层制动，不因三十秒存储上限崩溃。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stale_observation_keeps_its_real_age():
    observation = RuntimeLocalSafetyObservation(
        sequence=1,
        observed_at_unix_ms=0,
        source="onboard",
        stream_healthy=False,
        stream_age_seconds=120,
        localization_covariance_m2=0.1,
        current_position_m=_point(),
        current_velocity_mps=Vector3(x=0, y=0, z=0),
        target_position_m=_point(1),
    )
    request = LocalPlannerRequest(
        current_position_m=_point(),
        current_velocity_mps=Vector3(x=0, y=0, z=0),
        target_position_m=_point(1),
        vehicle_radius_m=0.2,
        vehicle_height_m=0.3,
        max_speed_mps=1,
        max_acceleration_mps2=1,
        required_clearance_m=0.2,
        prediction_horizon_seconds=3,
        prediction_step_seconds=0.1,
        perception_stream_healthy=False,
        perception_stream_age_seconds=observation.stream_age_seconds,
    )
    assert request.perception_stream_age_seconds == 120


# 功能：
#   验证安全命令不接受反向或过长有效期，即使其动作只是制动。
# 输入：
#   deadline：故意不符合短时命令要求的绝对期限，单位毫秒。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("deadline", [999, 4001])
def test_safety_command_has_a_bounded_horizon(deadline):
    decision = PredictiveSafetyDecision(
        action="hold",
        selected_velocity_mps=Vector3(x=0, y=0, z=0),
        control_source="deterministic-brake",
        predicted_path_m=[_point()],
        minimum_predicted_clearance_m=1,
        time_to_minimum_clearance_seconds=0,
        evaluated_candidate_count=1,
    )
    with pytest.raises(ValidationError, match="horizon"):
        RuntimeLocalSafetyCommand(
            observation_sha256="a" * 64,
            observation_sequence=1,
            generated_at_unix_ms=1000,
            valid_until_unix_ms=deadline,
            source="onboard",
            command_position_m=_point(),
            decision=decision,
        )


# 功能：
#   验证同名适配器不能通过重复声明规避执行器所有权检查。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_runtime_adapter_identity_is_unique():
    adapter = RuntimeActionAdapterDefinition(
        adapter_id="test.adapter",
        runtime_executors=["test.execute"],
        driver="ros2-service",
        authority="simulate",
    )
    with pytest.raises(ValidationError, match="unique"):
        RuntimeActionAdapterCatalog(
            catalog_id="runtime-actions." + "a" * 24,
            adapters=[adapter, adapter.model_copy(deep=True)],
        )


# 功能：
#   验证人工接管授权必须使用有时区的绝对时间，混合时间不能泄漏比较异常。
# 输入：
#   naive_expiry：是否仅将过期时间移除时区。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("naive_expiry", [False, True])
def test_takeover_requires_aware_absolute_time(naive_expiry):
    now = datetime.now(UTC)
    expires = (now + timedelta(seconds=2)).replace(tzinfo=None)
    with pytest.raises(ValidationError, match="timezone"):
        RuntimeOperatorTakeoverGrant(
            message_id="runtime-msg-" + "a" * 32,
            execution_id="execution-" + "a" * 32,
            operator_id="tester",
            message_sha256="a" * 64,
            hold_ack_sha256="a" * 64,
            decision_sha256="a" * 64,
            grant_token_sha256="a" * 64,
            maximum_horizontal_speed_mps=1,
            maximum_vertical_speed_mps=1,
            maximum_yaw_rate_dps=30,
            deterministic_gates={"verified": True},
            issued_at=now if naive_expiry else now.replace(tzinfo=None),
            expires_at=expires,
        )
