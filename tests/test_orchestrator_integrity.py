"""Offline preparation counterexamples; fixtures never demonstrate flight capability."""

from datetime import UTC, datetime
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from test_orchestrator_call_boundaries import _call, _harness, _Port
from test_orchestrator_validation import _contract, _map, _model_result, _task_graph
from test_planning_energy_gates import _vehicle

from dronedream_agent_core import orchestrator as module
from dronedream_agent_core.contracts import (
    ActionDefinition,
    DomainActionCatalog,
    GraphRoute,
    MapCatalog,
    MissionRequest,
    PlanCritique,
    Px4Track,
    TaskGraph,
    Vector3,
)
from dronedream_agent_core.model_harness.model_port import StructuredCallResult
from dronedream_agent_core.plugin_contracts import PluginSnapshot


# 功能：
#   构造明确属于离线测试的两点轨迹，世界 ENU 和执行器轴映射使用真实合同。
# 输入：
#   无。
# 输出：
#   fixtures：世界路线与对应轨迹的二元组。
def _track_fixture():
    route = GraphRoute(
        start_node="a",
        goal_node="b",
        node_ids=["a", "b"],
        edge_ids=["a-b"],
        positions_m=[Vector3(x=0, y=0, z=1), Vector3(x=1, y=0, z=1)],
        route_length_m=1,
        all_edges_flight_verified=False,
    )
    track = Px4Track(
        coordinate_contract={
            "model_root_world_enu_m": [0, 0, 0],
            "collision_center_offset_model_m": [0, 0, 0.2],
        },
        points=[
            {"x": 0, "y": 0, "z": 0.8, "phase": "launch", "speed_limit_mps": 0.5},
            {"x": 0, "y": 1, "z": 0.8, "phase": "transit", "speed_limit_mps": 0.5},
        ],
        source_world_points=[
            {"east_m": 0, "north_m": 0, "up_m": 1},
            {"east_m": 1, "north_m": 0, "up_m": 1},
        ],
        waypoint_hold_seconds=1,
    )
    fixtures = route, track
    return fixtures


# 功能：
#   构造不创建数据库、网络或插件进程的编排依赖，专测资源所有权与输入隔离。
# 输入：
#   tmp_path：测试目录。
# 输出：
#   dependencies：真实编排构造器需要的合同与借用依赖。
def _constructor_inputs(tmp_path):
    vehicle = _vehicle()
    dependencies = dict(
        config=module.PreparationConfig(provider="kimi", critic_provider="kimi"),
        map_catalog=MapCatalog(
            scene_id="fixture",
            semantic_sha256="a" * 64,
            entities=[
                {
                    "entity_id": "office",
                    "aliases": ["office"],
                    "position_m": {"x": 0, "y": 0, "z": 3},
                    "semantic": "launch",
                    "source_pointer": "fixture/office",
                }
            ],
            road_segment_ids=[],
            topology_available=True,
            known_limits=[],
        ),
        map_graph=_map(),
        semantic_path=tmp_path / "semantic.json",
        vehicle_sdf=tmp_path / "vehicle.sdf",
        vehicle_asset_id=vehicle.asset_id,
        vehicle=vehicle,
        context_store=object(),
        tool_registry=object(),
        extension_registry=object(),
        plugin_snapshot=PluginSnapshot(
            snapshot_id="plugin-snapshot-" + "a" * 24,
            catalog_sha256="b" * 64,
            plugins=[],
            created_at=datetime.now(UTC),
        ),
    )
    return dependencies


# 功能：
#   从任务夹具生成动作目录，验证目录存在并不意味着本次任务授权。
# 输入：
#   graph：离线任务图。
# 输出：
#   catalog：与任务证据、后备动作一致的动作目录。
def _action_catalog(graph):
    catalog = DomainActionCatalog(
        catalog_id="core.fixture",
        domain_ids=["core.flight"],
        actions=[
            ActionDefinition(
                action_id=task.action,
                domain_id="core.flight",
                label=task.action,
                description="Offline contract test action",
                movement=task.action in {"navigate", "return"},
                input_schema={"type": "object"},
                required_success_evidence=task.success_evidence,
                allowed_fallbacks=[task.fallback],
                simulator_executor="fixture." + task.action,
            )
            for task in graph.nodes
        ],
    )
    return catalog


# 功能：
#   验证已安装动作被本次合同排除后仍然不能执行，不能由目录恢复授权。
# 输入：
#   无。
# 输出：
#   None。
def test_catalog_does_not_expand_mission_authority():
    graph = _task_graph(include_stale_gate=False)
    contract = _contract()
    contract.authorized_actions.remove("pickup")
    with pytest.raises(module.MissionPreparationBlocked, match="UNAUTHORIZED_ACTION:pickup"):
        module._validate_task_graph(graph, contract, _map(), _action_catalog(graph))


# 功能：
#   验证异地验证、并行提前移动、重复取件和提前落地不能进入已验证计划。
# 输入：
#   defect：要施加的任务依赖或目标错误。
# 输出：
#   None。
@pytest.mark.parametrize(
    "defect", ["verification-location", "independent-move", "duplicate-pickup", "early-land"]
)
def test_task_order_and_location_are_causal(defect):
    graph = _task_graph(include_stale_gate=False)
    by_id = {task.task_id: task for task in graph.nodes}
    if defect == "verification-location":
        by_id["verify-recipient"].target_node = "office"
    elif defect == "independent-move":
        by_id["go-pickup"].depends_on = []
    elif defect == "duplicate-pickup":
        graph.nodes.append(by_id["pickup"].model_copy(update={"task_id": "second-pickup"}))
    else:
        by_id["land"].depends_on = ["takeoff"]
    with pytest.raises(module.MissionPreparationBlocked):
        module._validate_task_graph(graph, _contract(), _map())


# 功能：
#   验证同名公共节点但世界坐标不连续的路线不能拼接为可执行路线。
# 输入：
#   无。
# 输出：
#   None。
def test_route_join_requires_geometric_continuity():
    route, _ = _track_fixture()
    second = route.model_copy(
        deep=True,
        update={"start_node": "b", "goal_node": "c", "node_ids": ["b", "c"], "edge_ids": ["b-c"]},
    )
    with pytest.raises(module.MissionPreparationBlocked, match="DISCONTINUITY"):
        module._combine_routes([route, second])


# 功能：
#   验证跨边界重新检查模型内部状态，不把字符串布尔或篡改依赖当成合法合同。
# 输入：
#   无。
# 输出：
#   None。
def test_snapshot_rejects_bypassed_model_validation():
    _, track = _track_fixture()
    damaged = track.model_copy(update={"stop_at_waypoints": "false"})
    with pytest.raises(ValueError):
        module._artifact_snapshot(damaged, Px4Track)
    graph = _task_graph(include_stale_gate=False)
    graph.nodes[0].depends_on.append("land")
    with pytest.raises(ValueError, match="acyclic"):
        module._artifact_snapshot(graph, TaskGraph)


# 功能：
#   覆盖优化插件可能放宽的速度、停留、到达容差及稳定条件。
# 输入：
#   change：相对已导出轨迹的放宽修改。
# 输出：
#   None。
@pytest.mark.parametrize(
    "change",
    ["speed", "phase", "hold", "position", "speed-tolerance", "stable", "timeout", "coordinate"],
)
def test_track_optimizer_cannot_relax_the_exported_baseline(change):
    route, baseline = _track_fixture()
    track = baseline.model_copy(deep=True)
    if change == "speed":
        track.points[1].speed_limit_mps = 1  # Still below vehicle maximum; baseline must also bind.
    elif change == "phase":
        track.points[1].phase = "land"
    elif change == "hold":
        track.waypoint_hold_seconds = 0
    elif change == "position":
        track.waypoint_position_tolerance_m = 1
    elif change == "speed-tolerance":
        track.waypoint_speed_tolerance_mps = 1
    elif change == "stable":
        track.waypoint_stable_window_seconds = 0.1
    elif change == "timeout":
        track.waypoint_settle_timeout_seconds = 120
    else:
        track.coordinate_contract.model_root_world_enu_m[0] = 2
        for point in track.points:
            point.y -= 2  # Same reconstructed world geometry but different actuator frame.
    with pytest.raises(module.MissionPreparationBlocked):
        module._validate_plugin_track_tightening(track, route, _vehicle(), baseline=baseline)


# 功能：
#   验证合理降速、延长停留并收紧到达判定仍可通过，不将优化入口全部禁用。
# 输入：
#   无。
# 输出：
#   None。
def test_track_optimizer_can_tighten_constraints():
    route, baseline = _track_fixture()
    track = baseline.model_copy(deep=True)
    track.points[1].speed_limit_mps = 0.25
    track.waypoint_hold_seconds = 2
    track.waypoint_position_tolerance_m = 0.1
    track.waypoint_speed_tolerance_mps = 0.1
    track.waypoint_stable_window_seconds = 1
    module._validate_plugin_track_tightening(track, route, _vehicle(), baseline=baseline)


# 功能：
#   验证地图摘要保留完整未验证边数量，并保留返程再次经过同一区域的顺序。
# 输入：
#   无。
# 输出：
#   None。
def test_map_view_retains_complete_counts_and_return_sequence():
    context = {
        "topology": {
            "focus_routes": [
                {
                    "semantic_sequence": ["corridor", "stairs", "corridor"],
                    "unverified_edge_ids": ["one-of-many"],
                    "unverified_edge_count": 137,
                    "truncated_for_model_context": True,
                }
            ]
        }
    }
    view = module._semantic_planning_map_view(context, _contract())
    summary = view["topology"]["focus_route_summaries"][0]
    assert summary["unverified_edge_count"] == 137
    assert summary["semantic_sequence"] == ["corridor", "stairs", "corridor"]
    assert summary["semantic_sequence_truncated"] is True


# 功能：
#   覆盖可能被真值转换或过滤丢弃的门控输出，使错误值不能让计划通过。
# 输入：
#   result：非法或自相矛盾的插件验证材料。
# 输出：
#   None。
@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"accepted": "false"},
        {"accepted": 1},
        {"accepted": True, "deterministic_gates": {"clearance": False}},
        {"accepted": False, "issue_codes": "not-a-list"},
    ],
)
def test_invalid_plan_gate_is_not_an_implicit_accept(result):
    with pytest.raises(module.MissionPreparationBlocked, match="GATE_RESULT_INVALID"):
        module._rejected_plan_validators([result])


# 功能：
#   验证明确负载及悬停数值不能丢掉负号后成为看似正常的正值。
# 输入：
#   message：带错误物理数值的请求。
# 输出：
#   None。
@pytest.mark.parametrize("message", ["挂载-1公斤", "hover -20 seconds", "悬停0秒"])
def test_negative_user_values_are_not_reinterpreted(message):
    with pytest.raises(module.MissionPreparationBlocked, match="EXPLICIT_"):
        module._explicit_constraint_hints(
            MissionRequest(conversation_id="fixture", message=message)
        )


# 功能：
#   验证计量插件失败不会抹掉已经成功返回的供应商调用记录。
# 输入：
#   无。
# 输出：
#   None。
def test_meter_failure_keeps_the_provider_receipt():
    instance, _, evidence = _harness({"primary": _Port("offline", "1")}, ["primary"], {})

    # 功能：
    #   模拟计量钩子失败，不替换真实编排器的记录顺序。
    # 输入：
    #   args：调用插槽和钩子名。
    #   kwargs：原始调用记录等参数。
    # 输出：
    #   None。
    def fail_meter(*args, **kwargs):
        raise module.MissionPreparationBlocked("FIXTURE_METER_FAILURE")

    instance._invoke_multiple_extensions = fail_meter
    with pytest.raises(module.MissionPreparationBlocked, match="METER_FAILURE"):
        _call(instance, evidence)
    receipts = [value for name, value in evidence.events if name.endswith("provider-response")]
    assert len(receipts) == 1
    assert receipts[0]["record"]["input_tokens"] == 10


# 功能：
#   验证聚合结果不遗失支持响应内部已经附带的物理调用记录。
# 输入：
#   无。
# 输出：
#   None。
def test_nested_consensus_retains_all_physical_records():
    first = _model_result(PlanCritique(accepted=True), call_suffix="a")
    second = _model_result(PlanCritique(accepted=True), call_suffix="b")
    third = _model_result(PlanCritique(accepted=True), call_suffix="c")
    nested = StructuredCallResult(
        artifact=second.artifact, record=second.record, supporting_records=(third.record,)
    )
    result, _ = module._resolve_consensus_result(
        [first, nested], require_identical=True, record_dissent=True
    )
    assert [record.call_id for record in module._model_records(result)] == [
        first.record.call_id,
        second.record.call_id,
        third.record.call_id,
    ]


# 功能：
#   验证第二个端口构造失败时回收第一个已创建端口，依赖校验失败时不创建端口。
# 输入：
#   tmp_path：测试依赖路径。
#   monkeypatch：替换实际网络端口构造器。
# 输出：
#   None。
def test_constructor_cleans_owned_ports_on_partial_failure(tmp_path, monkeypatch):
    closed = []
    calls = []

    # 功能：
    #   首次返回可关闭连接夹具，第二次抛出初始化错误。
    # 输入：
    #   args：供应商参数。
    #   kwargs：重试与超时设置。
    # 输出：
    #   port：首个离线端口夹具。
    def build_port(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("FIXTURE_SECOND_PORT_FAILURE")
        port = SimpleNamespace(close=lambda: closed.append(True))
        return port

    monkeypatch.setattr(module, "StructuredModelPort", build_port)
    inputs = _constructor_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="SECOND_PORT_FAILURE"):
        module.MissionOrchestrator(**inputs)
    assert closed == [True]
    assert len(calls) == 2
    inputs["plugin_snapshot"] = None
    with pytest.raises(ValueError, match="supplied together"):
        module.MissionOrchestrator(**inputs)
    assert len(calls) == 2


# 功能：
#   验证借用端口不由编排器关闭，角色覆盖与属性一致，外部地图修改不污染已冻结依赖。
# 输入：
#   tmp_path：测试目录。
# 输出：
#   None。
def test_borrowed_ports_and_source_snapshots_remain_separate(tmp_path):
    inputs = _constructor_inputs(tmp_path)
    closed = []
    borrowed = SimpleNamespace(close=lambda: closed.append(True))
    instance = module.MissionOrchestrator(
        **inputs, primary_port=object(), critic_port=borrowed, model_ports={"primary": borrowed}
    )
    try:
        inputs["map_graph"].nodes[0].label = "external change"
        assert instance.map_graph.nodes[0].label != "external change"
        assert instance.primary is instance.model_ports["primary"] is borrowed
    finally:
        instance.close()
    instance.close()
    assert not closed
    with pytest.raises(module.MissionPreparationBlocked, match="CLOSED"):
        instance.prepare(MissionRequest(conversation_id="fixture", message="test task"), tmp_path)


# 功能：
#   验证同一个编排器不能同时准备两个任务，并在正常退出后恢复可关闭状态。
# 输入：
#   tmp_path：测试目录。
# 输出：
#   None。
def test_preparation_rejects_concurrent_reuse(tmp_path):
    borrowed = object()
    instance = module.MissionOrchestrator(
        **_constructor_inputs(tmp_path), primary_port=borrowed, critic_port=borrowed
    )
    entered, release = Event(), Event()
    results = []

    # 功能：
    #   阻塞内部准备方法以测试外层互斥，不声称运行了完整规划或飞行。
    # 输入：
    #   request：已冻结的请求。
    #   output_dir：测试输出路径。
    # 输出：
    #   message：供线程结果断言的输入消息。
    def delayed_prepare(request, output_dir):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("FIXTURE_TIMEOUT")
        message = request.message
        return message

    instance._prepare = delayed_prepare
    request = MissionRequest(conversation_id="fixture", message="original task")
    worker = Thread(target=lambda: results.append(instance.prepare(request, tmp_path)))
    worker.start()
    try:
        assert entered.wait(5)
        request.message = "mutated task"
        with pytest.raises(module.MissionPreparationBlocked, match="ALREADY_RUNNING"):
            instance.prepare(request, tmp_path)
        with pytest.raises(module.MissionPreparationBlocked, match="ALREADY_RUNNING"):
            instance.close()
    finally:
        release.set()
        worker.join(5)
        instance.close()
    assert not worker.is_alive()
    assert results == ["original task"]
