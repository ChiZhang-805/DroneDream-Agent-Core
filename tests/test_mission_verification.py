from __future__ import annotations

from types import SimpleNamespace

import pytest

from dronedream_agent_app.mission_service import _prepared_plan_gates
from dronedream_agent_core.contracts import (
    ActionDefinition,
    DomainActionCatalog,
    FlightPlan,
    GraphRoute,
    MissionContract,
    PlanSegment,
    Px4CoordinateContract,
    Px4Track,
    Px4TrackPoint,
    RouteClearanceReport,
    RoutePoint,
    RuntimeActionExecutionContract,
    RuntimeActionExecutionStep,
    RuntimeCheckpoint,
    RuntimeCheckpointContract,
    SemanticPlan,
    TaskGraph,
    TaskNode,
    Vector3,
    WorldTrackPoint,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.verification import (
    MissionVerificationPlanError,
    build_mission_verification_plan,
    verification_plan_matches_prepared_mission,
)


def _artifacts():
    """Build a fully bound round trip without a simulator or paid provider call."""
    actions = DomainActionCatalog(
        catalog_id="core.test-flight",
        domain_ids=["core.flight"],
        actions=[
            ActionDefinition(
                action_id="takeoff",
                domain_id="core.flight",
                label="Take off",
                description="Enter stable flight.",
                flight_boundary="takeoff",
                required_success_evidence=["airborne"],
                allowed_fallbacks=["hold"],
                simulator_executor="px4.takeoff",
                authority="control",
            ),
            ActionDefinition(
                action_id="navigate",
                domain_id="core.flight",
                label="Navigate",
                description="Fly to the target.",
                movement=True,
                required_success_evidence=["target reached"],
                allowed_fallbacks=["hold"],
                simulator_executor="px4.navigate",
                authority="control",
            ),
            ActionDefinition(
                action_id="return",
                domain_id="core.flight",
                label="Return",
                description="Fly to the launch point.",
                movement=True,
                required_success_evidence=["return node reached"],
                allowed_fallbacks=["land"],
                simulator_executor="px4.return",
                authority="control",
            ),
            ActionDefinition(
                action_id="land",
                domain_id="core.flight",
                label="Land",
                description="Land and make motors safe.",
                flight_boundary="landing",
                required_success_evidence=["landed", "motors safe"],
                allowed_fallbacks=["hold"],
                simulator_executor="px4.land",
                authority="control",
            ),
        ],
    )
    contract = MissionContract(
        contract_id="mission-0123456789abcdef01234567",
        conversation_id="thread-1",
        goal="visit target and return",
        start_node="office",
        target_node="target",
        return_node="office",
        payload_action="none",
        domain_ids=actions.domain_ids,
        authorized_actions=["takeoff", "navigate", "return", "land"],
        action_catalog_sha256=sha256_json(actions),
        map_asset_id="map",
        map_sha256="a" * 64,
        map_semantic_sha256="b" * 64,
        vehicle_asset_id="vehicle",
        vehicle_sha256="c" * 64,
        constraints=[],
        immutable_safety_rules=["model has no actuator authority"],
    )
    task_graph = TaskGraph(
        nodes=[
            TaskNode(
                task_id="takeoff",
                action="takeoff",
                target_node="office",
                success_evidence=["airborne"],
                fallback="hold",
            ),
            TaskNode(
                task_id="outbound",
                action="navigate",
                target_node="target",
                depends_on=["takeoff"],
                success_evidence=["target reached"],
                fallback="hold",
            ),
            TaskNode(
                task_id="return",
                action="return",
                target_node="office",
                depends_on=["outbound"],
                success_evidence=["return node reached"],
                fallback="land",
            ),
            TaskNode(
                task_id="land",
                action="land",
                target_node="office",
                depends_on=["return"],
                success_evidence=["landed", "motors safe"],
                fallback="hold",
            ),
        ]
    )
    semantic_plan = SemanticPlan(
        ordered_targets=["target", "office"], rationale_summary="Visit and return."
    )
    office = Vector3(x=0.0, y=0.0, z=1.0)
    target = Vector3(x=2.0, y=0.0, z=1.0)
    flight_plan = FlightPlan(
        revision=1,
        contract_id=contract.contract_id,
        semantic_plan_sha256=sha256_json(semantic_plan),
        segments=[
            PlanSegment(
                segment_id="segment-001",
                task_id="outbound",
                from_node="office",
                to_node="target",
                path=[
                    RoutePoint(node_id="office", position_m=office),
                    RoutePoint(node_id="target", position_m=target),
                ],
                speed_limit_mps=1.0,
                minimum_clearance_m=0.5,
                success_evidence=["target reached"],
            ),
            PlanSegment(
                segment_id="segment-002",
                task_id="return",
                from_node="target",
                to_node="office",
                path=[
                    RoutePoint(node_id="target", position_m=target),
                    RoutePoint(node_id="office", position_m=office),
                ],
                speed_limit_mps=1.0,
                minimum_clearance_m=0.5,
                success_evidence=["return node reached"],
            ),
        ],
    )
    route = GraphRoute(
        start_node="office",
        goal_node="office",
        node_ids=["office", "target", "office"],
        edge_ids=["outbound", "return"],
        positions_m=[office, target, office],
        route_length_m=4.0,
        all_edges_flight_verified=False,
    )
    clearance = RouteClearanceReport(
        accepted=True,
        route_sha256=sha256_json(route),
        semantic_sha256=contract.map_semantic_sha256,
        sample_interval_m=0.1,
        sample_count=41,
        primitive_count=1,
        collision_count=0,
        minimum_clearance_m=0.5,
        minimum_clearance_point=office,
        minimum_clearance_primitive="wall",
        segment_minimum_clearances_m=[0.5, 0.5],
    )
    track = Px4Track(
        coordinate_contract=Px4CoordinateContract(
            model_root_world_enu_m=[0.0, 0.0, 0.0],
            collision_center_offset_model_m=[0.0, 0.0, 0.3],
        ),
        points=[
            Px4TrackPoint(x=0, y=0, z=1, phase="launch", speed_limit_mps=1),
            Px4TrackPoint(x=0, y=2, z=1, phase="transit", speed_limit_mps=1),
            Px4TrackPoint(x=0, y=0, z=1, phase="land", speed_limit_mps=1),
        ],
        source_world_points=[
            WorldTrackPoint(east_m=0, north_m=0, up_m=1),
            WorldTrackPoint(east_m=2, north_m=0, up_m=1),
            WorldTrackPoint(east_m=0, north_m=0, up_m=1),
        ],
        waypoint_hold_seconds=0.4,
    )
    checkpoints = RuntimeCheckpointContract(
        contract_id=contract.contract_id,
        checkpoints=[
            RuntimeCheckpoint(
                checkpoint_id="checkpoint-001",
                segment_id="segment-001",
                task_id="outbound",
                track_point_index=1,
                target_node="target",
            ),
            RuntimeCheckpoint(
                checkpoint_id="checkpoint-002",
                segment_id="segment-002",
                task_id="return",
                track_point_index=2,
                target_node="office",
            ),
        ],
    )
    runtime_actions = RuntimeActionExecutionContract(
        contract_id=contract.contract_id,
        task_graph_sha256=sha256_json(task_graph),
        domain_action_catalog_sha256=sha256_json(actions),
        adapter_catalog_sha256="d" * 64,
        steps=[],
    )
    return (
        contract,
        actions,
        task_graph,
        semantic_plan,
        flight_plan,
        route,
        clearance,
        track,
        checkpoints,
        runtime_actions,
    )


def _build(values):
    """Exercise the product compiler with independently mutable artifact fixtures."""
    return build_mission_verification_plan(
        contract=values[0],
        domain_actions=values[1],
        task_graph=values[2],
        semantic_plan=values[3],
        flight_plan=values[4],
        execution_route=values[5],
        route_clearance=values[6],
        px4_track=values[7],
        runtime_checkpoints=values[8],
        runtime_actions=values[9],
    )


def test_verification_plan_covers_every_phase_and_binds_every_artifact() -> None:
    values = _artifacts()
    plan = _build(values)

    assert {item.phase for item in plan.requirements} == {
        "pre-confirmation",
        "runtime",
        "completion",
    }
    assert all(item.blocking for item in plan.requirements)
    assert not any(item.model_only_evidence_accepted for item in plan.requirements)
    assert verification_plan_matches_prepared_mission(
        verification_plan=plan,
        contract=values[0],
        domain_actions=values[1],
        task_graph=values[2],
        semantic_plan=values[3],
        flight_plan=values[4],
        execution_route=values[5],
        route_clearance=values[6],
        px4_track=values[7],
        runtime_checkpoints=values[8],
        runtime_actions=values[9],
    )


def test_verification_plan_rejects_semantically_negated_task_evidence() -> None:
    values = list(_artifacts())
    values[2] = values[2].model_copy(deep=True)
    values[2].nodes[0].success_evidence = ["not airborne"]
    values[9] = values[9].model_copy(update={"task_graph_sha256": sha256_json(values[2])})

    with pytest.raises(
        MissionVerificationPlanError, match="task_evidence_matches_catalog"
    ):
        _build(values)


def test_execution_binding_rejects_a_verification_plan_with_an_omitted_task_obligation() -> None:
    values = _artifacts()
    plan = _build(values)
    reduced = plan.model_copy(
        update={
            "requirements": [
                requirement
                for requirement in plan.requirements
                if not (
                    requirement.phase == "runtime"
                    and requirement.subject_id == "outbound"
                )
            ]
        }
    )

    assert not verification_plan_matches_prepared_mission(
        verification_plan=reduced,
        contract=values[0],
        domain_actions=values[1],
        task_graph=values[2],
        semantic_plan=values[3],
        flight_plan=values[4],
        execution_route=values[5],
        route_clearance=values[6],
        px4_track=values[7],
        runtime_checkpoints=values[8],
        runtime_actions=values[9],
    )


def test_public_feasibility_uses_continuous_clearance_not_historical_edge_labels() -> None:
    values = _artifacts()
    prepared = SimpleNamespace(
        intent_critique=SimpleNamespace(accepted=True),
        plan_critique=SimpleNamespace(accepted=True),
        route_clearance=values[6],
        execution_route=values[5],
        contract=values[0],
        verification_plan=None,
    )

    assert values[5].all_edges_flight_verified is False
    assert all(_prepared_plan_gates(prepared).values())


def _action_artifacts():
    """Add one non-motion action so duplicate and swapped step bindings are observable."""
    values = list(_artifacts())
    for action_id in ("photo", "inspect"):
        values[1].actions.append(ActionDefinition(
            action_id=action_id, domain_id="core.flight", label=action_id,
            description="Observe target without granting navigation authority.",
            runtime_executor=f"native.{action_id}", simulator_executor=f"native.{action_id}",
            required_success_evidence=[f"{action_id} observed"],
            allowed_fallbacks=["hold"], authority="control",
        ))
    values[0].authorized_actions.extend(["photo", "inspect"])
    values[0].action_catalog_sha256 = sha256_json(values[1])
    values[2].nodes.append(TaskNode(
        task_id="photo", action="photo", target_node="target", depends_on=["outbound"],
        success_evidence=["photo observed"], fallback="hold",
    ))
    values[9] = values[9].model_copy(update={
        "task_graph_sha256": sha256_json(values[2]),
        "domain_action_catalog_sha256": sha256_json(values[1]),
        "steps": [RuntimeActionExecutionStep(
            step_id="action-001", task_id="photo", action="photo", target_node="target",
            trigger="checkpoint", checkpoint_id="checkpoint-001",
            runtime_executor="native.photo", adapter_id="camera", driver="mavsdk-camera",
            required_success_evidence=["photo observed"], max_attempts=1, fallback="hold",
            timeout_seconds=1, authority="control",
        )],
    })
    return values


def test_runtime_action_verification_accepts_exact_task_binding():
    assert _build(_action_artifacts()).deterministic_gates


@pytest.mark.parametrize("change", [
    {"action": "inspect", "required_success_evidence": ["inspect observed"],
     "runtime_executor": "native.inspect"},
    {"target_node": "office"}, {"runtime_executor": "native.inspect"},
    {"fallback": "land"},
])
def test_runtime_action_verification_rejects_swapped_execution_semantics(change):
    values = _action_artifacts()
    values[9].steps[0] = values[9].steps[0].model_copy(update=change)
    with pytest.raises(MissionVerificationPlanError, match="runtime_action_steps_match_tasks"):
        _build(values)


def test_runtime_action_verification_rejects_duplicate_task_before_dictionary_collapse():
    values = _action_artifacts()
    values[9].steps.append(values[9].steps[0].model_copy(update={"step_id": "action-002"}))
    with pytest.raises(MissionVerificationPlanError, match="runtime_action_steps_unique"):
        _build(values)


def test_verification_rejects_clearance_for_a_different_map():
    values = list(_artifacts())
    values[6] = values[6].model_copy(update={"semantic_sha256": "f" * 64})
    with pytest.raises(MissionVerificationPlanError, match="clearance_bound_to_mission_map"):
        _build(values)
