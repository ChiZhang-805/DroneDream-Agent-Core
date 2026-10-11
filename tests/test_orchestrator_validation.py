from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from dronedream_agent_core.contracts import (
    ActionDefinition,
    DomainActionCatalog,
    GraphRoute,
    IntentArtifact,
    IntentCritique,
    MapAsset,
    MapCatalog,
    MissionContract,
    MissionRequest,
    ModelCallRecord,
    PlanCritique,
    RouteClearanceReport,
    SemanticPlan,
    TaskGraph,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.model_harness.model_port import StructuredCallResult
from dronedream_agent_core.orchestrator import (
    MissionPreparationBlocked,
    _bind_explicit_intent_constraints,
    _bind_product_intent_defaults,
    _bind_task_graph_action_contracts,
    _decomposition_action_catalog_view,
    _explicit_constraint_hints,
    _ground_intent_clarification_options,
    _intent_action_catalog_view,
    _intent_map_catalog_view,
    _intent_repair_feedback,
    _intent_workflow_scope,
    _missing_explicit_constraints,
    _model_records,
    _normalize_intent_critique_scope,
    _recommended_plugin_tools,
    _resolve_consensus_result,
    _route_objective_weights,
    _route_operational_clearance_assessment,
    _route_recovery_clarification,
    _semantic_planning_map_view,
    _validate_explicit_payload_capacity,
    _validate_semantic_plan,
    _validate_task_graph,
)


def _model_result(
    artifact: PlanCritique | SemanticPlan, *, call_suffix: str
) -> StructuredCallResult:
    return StructuredCallResult(
        artifact=artifact,
        record=ModelCallRecord(
            call_id=f"model-{call_suffix * 24}",
            role="plan_critic",
            attempt=1,
            input_sha256="1" * 64,
            output_sha256=sha256_json(artifact),
            output_schema=type(artifact).__name__,
            provider=f"provider-{call_suffix}",
            model=f"model-{call_suffix}",
            latency_ms=10,
            created_at=datetime.now(UTC),
        ),
    )


def test_intent_clarification_options_are_ordered_and_bounded() -> None:
    intent = IntentArtifact(
        goal="从办公室取餐",
        start_entity="launch",
        target_entity="__unresolved__",
        return_entity="launch",
        payload_action="pickup",
        missing_critical_fields=["请选择取餐点"],
        clarification_questions=[
            {
                "question_id": "pickup.location",
                "prompt": "你希望去哪个取餐点？",
                "options": [
                    {"option_id": "A", "label": "食品储藏室", "response": "去食品储藏室取餐。"},
                    {"option_id": "B", "label": "补给站", "response": "去补给站取餐。"},
                ],
                "allow_other": True,
                "other_label": "其他",
            }
        ],
    )
    assert intent.clarification_questions[0].options[0].option_id == "A"

    with pytest.raises(ValueError, match="ordered A/B/C"):
        IntentArtifact(
            goal="从办公室取餐",
            start_entity="launch",
            target_entity="__unresolved__",
            return_entity="launch",
            payload_action="pickup",
            missing_critical_fields=["请选择取餐点"],
            clarification_questions=[
                {
                    "question_id": "pickup.location",
                    "prompt": "你希望去哪个取餐点？",
                    "options": [
                        {"option_id": "B", "label": "补给站", "response": "去补给站取餐。"},
                        {"option_id": "A", "label": "食品储藏室", "response": "去食品储藏室取餐。"},
                    ],
                }
            ],
        )


def test_clarification_options_bind_one_entity_and_reject_grouped_choice() -> None:
    graph = _map()
    catalog = MapCatalog(
        scene_id="office",
        semantic_sha256="a" * 64,
        topology_available=True,
        entities=[
            {
                "entity_id": "office",
                "aliases": ["办公室", "office"],
                "position_m": {"x": 0, "y": 0, "z": 3},
                "semantic": "launch",
                "source_pointer": "test:office",
            },
            {
                "entity_id": "pickup",
                "aliases": ["物资区", "supplies"],
                "position_m": {"x": 10, "y": 0, "z": 1},
                "semantic": "pickup",
                "source_pointer": "test:pickup",
            },
        ],
    )
    intent = IntentArtifact(
        goal="帮我取外卖",
        start_entity="office",
        target_entity="__unresolved__",
        return_entity="office",
        payload_action="pickup",
        missing_critical_fields=["请选择取餐点"],
        clarification_questions=[
            {
                "question_id": "pickup.location",
                "prompt": "外卖放在哪里？",
                "options": [
                    {"option_id": "A", "label": "物资区", "response": "去物资区取外卖"},
                    {
                        "option_id": "B",
                        "label": "办公室 / 物资区",
                        "response": "去 office 或 pickup 取外卖",
                    },
                ],
            }
        ],
    )

    grounded, issues = _ground_intent_clarification_options(intent, catalog, graph)

    assert grounded.clarification_questions[0].options[0].response.endswith("（pickup）。")
    assert "CLARIFICATION_OPTION_ENTITY_NOT_UNIQUE" in issues


def test_route_exhaustion_keeps_semantic_places_without_authored_edges() -> None:
    graph = MapAsset.model_validate(
        {
            "asset_id": "office-map",
            "name": "Office Map",
            "nodes": [
                {
                    "node_id": "office",
                    "label": "Office",
                    "position_m": {"x": 0, "y": 0, "z": 1},
                    "semantic": "launch",
                },
                {
                    "node_id": "blocked",
                    "label": "Blocked",
                    "position_m": {"x": 3, "y": 0, "z": 1},
                    "semantic": "pickup",
                },
                {
                    "node_id": "pantry",
                    "label": "Pantry",
                    "position_m": {"x": 5, "y": 0, "z": 1},
                    "semantic": "pickup",
                },
                {
                    "node_id": "supplies",
                    "label": "Supplies",
                    "position_m": {"x": 7, "y": 0, "z": 1},
                    "semantic": "pickup",
                },
                {
                    "node_id": "isolated",
                    "label": "Isolated",
                    "position_m": {"x": 9, "y": 0, "z": 1},
                    "semantic": "pickup",
                },
            ],
            "edges": [
                {
                    "edge_id": "office-blocked",
                    "from_node": "office",
                    "to_node": "blocked",
                    "distance_m": 3,
                    "minimum_clearance_m": 1,
                    "speed_limit_mps": 1,
                },
                {
                    "edge_id": "office-pantry",
                    "from_node": "office",
                    "to_node": "pantry",
                    "distance_m": 5,
                    "minimum_clearance_m": 1,
                    "speed_limit_mps": 1,
                },
                {
                    "edge_id": "office-supplies",
                    "from_node": "office",
                    "to_node": "supplies",
                    "distance_m": 7,
                    "minimum_clearance_m": 1,
                    "speed_limit_mps": 1,
                },
            ],
            "named_entities": {
                "office": "office",
                "blocked": "blocked",
                "pantry": "pantry",
                "supplies": "supplies",
                "isolated": "isolated",
            },
        }
    )
    catalog = MapCatalog(
        scene_id="office",
        semantic_sha256="a" * 64,
        topology_available=True,
        entities=[
            {
                "entity_id": node_id,
                "aliases": aliases,
                "position_m": position,
                "semantic": semantic,
                "source_pointer": f"test:{node_id}",
            }
            for node_id, aliases, position, semantic in [
                ("office", ["办公室", "office"], {"x": 0, "y": 0, "z": 1}, "launch"),
                ("blocked", ["封闭取餐点", "blocked"], {"x": 3, "y": 0, "z": 1}, "pickup"),
                ("pantry", ["茶水间", "pantry"], {"x": 5, "y": 0, "z": 1}, "pickup"),
                ("supplies", ["物资区", "supplies"], {"x": 7, "y": 0, "z": 1}, "pickup"),
                ("isolated", ["隔离区", "isolated"], {"x": 9, "y": 0, "z": 1}, "pickup"),
            ]
        ],
    )
    contract = _contract().model_copy(
        update={
            "map_asset_id": "office-map",
            "target_node": "blocked",
        }
    )
    intent = IntentArtifact(
        goal="帮我取一下外卖",
        start_entity="office",
        target_entity="blocked",
        return_entity="office",
        payload_action="pickup",
    )

    fields, questions = _route_recovery_clarification(
        MissionRequest(conversation_id="thread-route-recovery", message="帮我取一下外卖"),
        intent,
        contract,
        catalog,
        graph,
    )

    assert fields == ["当前目的地没有满足飞行条件的路线，请选择另一个地图地点"]
    assert [option["option_id"] for option in questions[0]["options"]] == ["A", "B", "C"]
    assert [option["label"] for option in questions[0]["options"]] == [
        "隔离区 (isolated)",
        "茶水间 (pantry)",
        "物资区 (supplies)",
    ]
    assert "blocked" not in str(questions)
    assert "isolated" in str(questions)


def test_pickup_product_defaults_return_to_start_and_add_bounded_hover() -> None:
    intent = IntentArtifact(
        goal="帮我取一下外卖",
        start_entity="launch",
        target_entity="pantry",
        return_entity="pantry",
        payload_action="pickup",
    )
    request = MissionRequest(
        conversation_id="conversation-test",
        message="帮我取一下外卖",
        start_entity="launch",
    )

    resolved = _bind_product_intent_defaults(intent, request)

    assert resolved.return_entity == "launch"
    assert "pickup_hover_seconds=10" in resolved.constraints


def test_pickup_product_defaults_preserve_explicit_one_way_destination() -> None:
    intent = IntentArtifact(
        goal="去餐点取件并在那里降落",
        start_entity="launch",
        target_entity="pantry",
        return_entity="pantry",
        payload_action="pickup",
    )
    request = MissionRequest(
        conversation_id="conversation-test",
        message="去餐点取件并在那里降落，不用回来",
        start_entity="launch",
    )

    assert _bind_product_intent_defaults(intent, request).return_entity == "pantry"


def test_delivery_product_defaults_return_to_start_without_requiring_user_to_repeat_it() -> None:
    intent = IntentArtifact(
        goal="把样品送到化学楼",
        start_entity="launch",
        target_entity="chemicum-delivery",
        return_entity="chemicum-delivery",
        payload_action="delivery.release-payload",
    )
    request = MissionRequest(
        conversation_id="conversation-delivery",
        message="把样品送到化学楼",
        start_entity="launch",
    )

    resolved = _bind_product_intent_defaults(intent, request)

    assert resolved.return_entity == "launch"
    assert "pickup_hover_seconds=10" not in resolved.constraints


def test_task_graph_requires_the_contract_delivery_action_at_the_target() -> None:
    contract = _contract().model_copy(
        update={
            "payload_action": "delivery.release-payload",
            "authorized_actions": [
                *_contract().authorized_actions,
                "delivery.release-payload",
                "delivery.verify-release-area",
            ],
        }
    )

    with pytest.raises(
        MissionPreparationBlocked,
        match="TASK_GRAPH_MISSING_PAYLOAD_ACTION:delivery.release-payload",
    ):
        _validate_task_graph(_task_graph(include_stale_gate=False), contract, _map())


def test_dual_review_consensus_uses_the_conservative_rejection_and_keeps_every_record() -> None:
    accepted = _model_result(PlanCritique(accepted=True), call_suffix="a")
    rejected = _model_result(
        PlanCritique(
            accepted=False,
            issue_codes=["CLEARANCE_EVIDENCE_MISSING"],
            repair_instructions=["Re-run the qualified clearance tool."],
        ),
        call_suffix="b",
    )

    resolved, resolution = _resolve_consensus_result(
        [accepted, rejected], require_identical=False, record_dissent=True
    )

    assert resolved.artifact == rejected.artifact
    assert resolution == "conservative-rejection"
    assert [record.call_id for record in _model_records(resolved)] == [
        rejected.record.call_id,
        accepted.record.call_id,
    ]


def test_non_review_consensus_dissent_fails_closed_instead_of_discarding_a_call() -> None:
    first = _model_result(
        SemanticPlan(ordered_targets=["pickup"], rationale_summary="Direct."),
        call_suffix="a",
    )
    second = _model_result(
        SemanticPlan(ordered_targets=["office"], rationale_summary="Return."),
        call_suffix="b",
    )

    with pytest.raises(MissionPreparationBlocked, match="MODEL_CONSENSUS_DISSENT_UNRESOLVED"):
        _resolve_consensus_result([first, second], require_identical=False, record_dissent=True)


def test_semantic_planning_map_view_removes_raw_geometry_and_irrelevant_entities() -> None:
    context = {
        "source_of_truth": "qualified-structured-map-not-rendered-image",
        "graph_sha256": "1" * 64,
        "semantic_sha256": "2" * 64,
        "coordinate_frame": "ENU",
        "bounds_m": {"minimum": {"x": 0}, "maximum": {"x": 10}},
        "topology": {
            "complete_graph_node_count": 3,
            "complete_graph_edge_count": 2,
            "truncated_for_model_context": False,
            "focus_nodes": ["office", "pickup"],
            "focus_routes": [
                {
                    "start_node": "office",
                    "goal_node": "pickup",
                    "node_ids": ["office", "middle", "pickup"],
                    "edge_ids": ["edge-a", "edge-b"],
                    "route_length_m": 10.0,
                    "minimum_clearance_m": 0.4,
                    "minimum_speed_limit_mps": 0.5,
                    "semantic_sequence": ["launch", "corridor", "pickup"],
                    "unverified_edge_ids": ["edge-b"],
                    "candidate_route_count": 1,
                    "candidate_routes": [
                        {
                            "node_ids": ["office", "middle", "pickup"],
                            "route_length_m": 10.0,
                            "minimum_clearance_m": 0.4,
                            "minimum_speed_limit_mps": 0.5,
                            "all_edges_flight_verified": False,
                            "route_sha256": "3" * 64,
                        }
                    ],
                }
            ],
        },
        "named_entities": [
            {"entity_id": "office", "node_id": "office"},
            {"entity_id": "pickup", "node_id": "pickup"},
            {"entity_id": "unrelated", "node_id": "stale-gate"},
        ],
        "nodes": [{"node_id": "middle", "position_m": {"x": 5, "y": 0, "z": 1}}],
        "edges": [{"edge_id": "edge-a"}],
        "semantic_environment": {"occupancy_ready": True},
        "catalog_known_limits": ["dynamic people require runtime sensing"],
        "model_authority": {"may_reason_about": ["risk-flags"]},
    }

    view = _semantic_planning_map_view(context, _contract())

    encoded = json.dumps(view, sort_keys=True)
    assert "node_ids" not in encoded
    assert "edge_ids" not in encoded
    assert "position_m" not in encoded
    assert "stale-gate" not in encoded
    assert view["source_context_sha256"] == sha256_json(context)
    route = view["topology"]["focus_route_summaries"][0]
    assert route["unverified_edge_count"] == 1
    assert route["semantic_sequence"] == ["launch", "corridor", "pickup"]
    assert route["geometry_authority"] == "deterministic-selected-route-tools"
    assert route["continuous_clearance_evaluated_here"] is False
    assert "minimum_clearance_m" not in route
    assert "all_edges_flight_verified" not in route
    assert all(
        "minimum_clearance_m" not in candidate and "all_edges_flight_verified" not in candidate
        for candidate in route["candidate_summaries"]
    )


def test_semantic_plan_bounds_explanatory_prose_without_changing_control_fields() -> None:
    plan = SemanticPlan.model_validate(
        {
            "ordered_targets": ["pickup", "office"],
            "route_policy": "clearance-first",
            "rationale_summary": "x" * 900,
        }
    )

    assert plan.ordered_targets == ["pickup", "office"]
    assert plan.route_policy == "clearance-first"
    assert plan.rationale_summary == "x" * 600


def _clearance_route() -> GraphRoute:
    return GraphRoute(
        start_node="start",
        goal_node="goal",
        node_ids=["start", "middle", "goal"],
        edge_ids=["edge-a", "edge-b"],
        positions_m=[
            Vector3(x=0.0, y=0.0, z=1.0),
            Vector3(x=1.0, y=0.0, z=1.0),
            Vector3(x=2.0, y=0.0, z=1.0),
        ],
        route_length_m=2.0,
        all_edges_flight_verified=False,
    )


def _clearance_report(segment_clearances_m: list[float]) -> RouteClearanceReport:
    minimum = min(segment_clearances_m)
    return RouteClearanceReport(
        accepted=True,
        route_sha256="a" * 64,
        semantic_sha256="b" * 64,
        sample_interval_m=0.1,
        sample_count=21,
        primitive_count=1,
        collision_count=0,
        minimum_clearance_m=minimum,
        minimum_clearance_point=Vector3(x=0.5, y=0.0, z=1.0),
        minimum_clearance_primitive="wall",
        segment_minimum_clearances_m=segment_clearances_m,
    )


def test_narrow_collision_free_segment_is_budgeted_for_precision_control() -> None:
    assessment = _route_operational_clearance_assessment(
        _clearance_route(),
        _clearance_report([0.2162, 0.4]),
        preferred_transit_clearance_m=0.35,
    )

    assert assessment["accepted"] is True
    assert assessment["topology_matches"] is True
    assert assessment["segment_budget_count"] == 2
    assert assessment["precision_control_required"] is True
    assert assessment["precision_segment_indexes"] == [0]
    assert assessment["rejected_segment_indexes"] == []


def test_segment_that_cannot_fund_runtime_margins_is_rejected() -> None:
    assessment = _route_operational_clearance_assessment(
        _clearance_route(),
        _clearance_report([0.10, 0.4]),
        preferred_transit_clearance_m=0.35,
    )

    assert assessment["accepted"] is False
    assert assessment["rejected_segment_indexes"] == [0]


def test_clearance_segment_topology_mismatch_is_rejected() -> None:
    assessment = _route_operational_clearance_assessment(
        _clearance_route(),
        _clearance_report([0.4]),
        preferred_transit_clearance_m=0.35,
    )

    assert assessment["accepted"] is False
    assert assessment["topology_matches"] is False


def test_clearance_first_policy_materially_increases_clearance_priority() -> None:
    balanced = _route_objective_weights("balanced")
    clearance_first = _route_objective_weights("clearance-first")

    assert clearance_first["minimum_clearance_m"] > balanced["minimum_clearance_m"]
    assert clearance_first["distance_m"] < balanced["distance_m"]
    assert sum(clearance_first.values()) == pytest.approx(1.0)


def test_unknown_route_policy_is_rejected() -> None:
    with pytest.raises(MissionPreparationBlocked, match="SEMANTIC_ROUTE_POLICY_INVALID"):
        _route_objective_weights("invented")


def _map() -> MapAsset:
    return MapAsset.model_validate(
        {
            "asset_id": "school-map",
            "name": "School Map",
            "nodes": [
                {
                    "node_id": "office",
                    "label": "Office",
                    "position_m": {"x": 0, "y": 0, "z": 3},
                    "semantic": "launch",
                },
                {
                    "node_id": "pickup",
                    "label": "Pickup",
                    "position_m": {"x": 10, "y": 0, "z": 1},
                    "semantic": "pickup",
                },
                {
                    "node_id": "stale-gate",
                    "label": "Old destination",
                    "position_m": {"x": 5, "y": 5, "z": 1},
                    "semantic": "outdoor",
                },
            ],
            "edges": [
                {
                    "edge_id": "office-pickup",
                    "from_node": "office",
                    "to_node": "pickup",
                    "distance_m": 10,
                    "minimum_clearance_m": 1,
                    "speed_limit_mps": 1,
                },
                {
                    "edge_id": "office-gate",
                    "from_node": "office",
                    "to_node": "stale-gate",
                    "distance_m": 7,
                    "minimum_clearance_m": 1,
                    "speed_limit_mps": 1,
                },
            ],
            "named_entities": {"office": "office", "pickup": "pickup"},
        }
    )


def _contract() -> MissionContract:
    digest = "a" * 64
    return MissionContract(
        contract_id="mission-0123456789abcdef01234567",
        conversation_id="thread-1",
        goal="retrieve parcel and return",
        start_node="office",
        target_node="pickup",
        return_node="office",
        payload_action="pickup",
        map_asset_id="school-map",
        map_sha256=digest,
        map_semantic_sha256=digest,
        vehicle_asset_id="my-drone",
        vehicle_sha256=digest,
        constraints=[],
        authorized_actions=[
            "takeoff",
            "traverse",
            "navigate",
            "delivery.verify-recipient",
            "pickup",
            "return",
            "land",
        ],
        immutable_safety_rules=["model has no actuator authority"],
    )


def _task_graph(*, include_stale_gate: bool) -> TaskGraph:
    nodes = [
        {
            "task_id": "takeoff",
            "action": "takeoff",
            "target_node": "office",
            "success_evidence": ["airborne"],
            "fallback": "abort",
        }
    ]
    if include_stale_gate:
        nodes.append(
            {
                "task_id": "old-destination",
                "action": "traverse",
                "target_node": "stale-gate",
                "depends_on": ["takeoff"],
                "success_evidence": ["arrived"],
                "fallback": "return",
            }
        )
    dependency = "old-destination" if include_stale_gate else "takeoff"
    nodes.extend(
        [
            {
                "task_id": "go-pickup",
                "action": "navigate",
                "target_node": "pickup",
                "depends_on": [dependency],
                "success_evidence": ["arrived"],
                "fallback": "return",
            },
            {
                "task_id": "verify-recipient",
                "action": "delivery.verify-recipient",
                "target_node": "pickup",
                "depends_on": ["go-pickup"],
                "success_evidence": ["recipient identity verified"],
                "fallback": "return",
            },
            {
                "task_id": "pickup",
                "action": "pickup",
                "target_node": "pickup",
                "depends_on": ["verify-recipient"],
                "success_evidence": ["payload verified"],
                "fallback": "return",
            },
            {
                "task_id": "return",
                "action": "return",
                "target_node": "office",
                "depends_on": ["pickup"],
                "success_evidence": ["office reached"],
                "fallback": "land",
            },
            {
                "task_id": "land",
                "action": "land",
                "target_node": "office",
                "depends_on": ["return"],
                "success_evidence": ["landed"],
                "fallback": "abort",
            },
        ]
    )
    return TaskGraph(nodes=nodes)


def test_task_graph_rejects_destination_left_over_from_superseded_plan() -> None:
    with pytest.raises(MissionPreparationBlocked, match="TASK_GRAPH_UNAUTHORIZED_MOVEMENT_TARGET"):
        _validate_task_graph(_task_graph(include_stale_gate=True), _contract(), _map())


def test_task_graph_accepts_only_contract_movement_targets() -> None:
    _validate_task_graph(_task_graph(include_stale_gate=False), _contract(), _map())


def test_intent_action_catalog_omits_execution_and_verification_contracts() -> None:
    catalog = DomainActionCatalog(
        catalog_id="core.intent-test",
        domain_ids=["core.flight"],
        actions=[
            ActionDefinition(
                action_id=action_id,
                domain_id="core.flight",
                label=action_id,
                description=f"Classify {action_id} intent.",
                payload=action_id == "pickup",
                flight_boundary=(
                    "takeoff"
                    if action_id == "takeoff"
                    else "landing"
                    if action_id == "land"
                    else "none"
                ),
                input_schema={"type": "object", "properties": {"secret": {"type": "string"}}},
                required_success_evidence=[f"{action_id} observed"],
                allowed_fallbacks=["abort"],
                simulator_executor=f"sim.test.{action_id}",
                runtime_executor=f"runtime.test.{action_id}",
                authority="control",
            )
            for action_id in ("takeoff", "pickup", "land")
        ],
    )

    view = _intent_action_catalog_view(catalog)

    assert [item["action_id"] for item in view["actions"]] == [
        "takeoff",
        "pickup",
        "land",
    ]
    serialized = json.dumps(view)
    assert "input_schema" not in serialized
    assert "runtime_executor" not in serialized
    assert "required_success_evidence" not in serialized
    assert "allowed_fallbacks" not in serialized


def test_intent_map_catalog_keeps_identity_but_omits_coordinates_and_source_paths() -> None:
    catalog = MapCatalog.model_validate(
        {
            "scene_id": "school",
            "semantic_sha256": "a" * 64,
            "entities": [
                {
                    "entity_id": "office-launch-pad",
                    "aliases": ["office", "办公室"],
                    "position_m": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "semantic": "launch",
                    "source_pointer": "/world/model[1]",
                }
            ],
            "road_segment_ids": ["road-1", "road-2"],
            "topology_available": True,
            "known_limits": ["dynamic obstacles require live sensing"],
        }
    )

    view = _intent_map_catalog_view(catalog)
    serialized = json.dumps(view, ensure_ascii=False)

    assert view["entity_count"] == 1
    assert view["road_segment_count"] == 2
    assert view["entities"][0]["aliases"] == ["office", "办公室"]
    assert "position_m" not in serialized
    assert "source_pointer" not in serialized


def test_core_binds_explicit_constraints_without_overwriting_model_interpretation() -> None:
    intent = IntentArtifact(
        goal="Retrieve the parcel and return.",
        start_entity="office-launch-pad",
        target_entity="takeout-pickup-pad",
        return_entity="office-launch-pad",
        payload_action="pickup",
        constraints=["plan_only"],
    )

    bound = _bind_explicit_intent_constraints(
        intent,
        ["plan_only", "do_not_execute", "return_to_start"],
    )

    assert bound.constraints == ["plan_only", "do_not_execute", "return_to_start"]
    assert bound.goal == intent.goal
    assert bound.target_entity == intent.target_entity


def test_workflow_scope_supports_round_trips_without_granting_hardware_authority() -> None:
    scope = _intent_workflow_scope()

    assert scope["round_trip_supported"] is True
    assert scope["physical_hardware_authority"] is False
    assert "one_way_return_entity_semantics" not in scope
    assert "round-trip" in str(scope["return_entity_semantics"])


def test_decomposition_view_and_binding_keep_runtime_evidence_core_owned() -> None:
    graph = _task_graph(include_stale_gate=False)
    actions = []
    for task in graph.nodes:
        actions.append(
            ActionDefinition(
                action_id=task.action,
                domain_id="core.flight",
                label=task.action,
                description=f"Task action {task.action}.",
                movement=task.action in {"navigate", "return"},
                flight_boundary=(
                    "takeoff"
                    if task.action == "takeoff"
                    else "landing"
                    if task.action == "land"
                    else "none"
                ),
                input_schema={"type": "object"},
                required_success_evidence=[f"canonical {task.action} evidence"],
                allowed_fallbacks=[task.fallback],
                simulator_executor=f"sim.test.{task.action}",
                runtime_executor=f"runtime.test.{task.action}",
            )
        )
    catalog = DomainActionCatalog(
        catalog_id="core.decomposition-test",
        domain_ids=["core.flight"],
        actions=actions,
    )

    view = _decomposition_action_catalog_view(catalog)
    bound = _bind_task_graph_action_contracts(graph, catalog)

    assert "required_success_evidence" not in json.dumps(view)
    assert all(
        task.success_evidence == [f"canonical {task.action} evidence"] for task in bound.nodes
    )
    assert [task.depends_on for task in bound.nodes] == [task.depends_on for task in graph.nodes]


def test_task_graph_requires_exact_catalog_evidence_not_a_negated_substring() -> None:
    graph = _task_graph(include_stale_gate=False)
    graph.nodes[0] = graph.nodes[0].model_copy(update={"success_evidence": ["not airborne"]})
    catalog = DomainActionCatalog(
        catalog_id="core.test",
        domain_ids=["core.flight"],
        actions=[
            ActionDefinition(
                action_id=task.action,
                domain_id="core.flight",
                label=task.action,
                description=f"Test definition for {task.action}.",
                movement=task.action in {"traverse", "navigate", "return"},
                flight_boundary=(
                    "takeoff"
                    if task.action == "takeoff"
                    else "landing"
                    if task.action == "land"
                    else "none"
                ),
                required_success_evidence=[
                    {
                        "takeoff": "airborne",
                        "navigate": "arrived",
                        "delivery.verify-recipient": "recipient identity verified",
                        "pickup": "payload verified",
                        "return": "office reached",
                        "land": "landed",
                    }[task.action]
                ],
                allowed_fallbacks=[task.fallback],
                simulator_executor=f"sim.test.{task.action}",
            )
            for task in graph.nodes
        ],
    )

    with pytest.raises(MissionPreparationBlocked, match="TASK_GRAPH_ACTION_EVIDENCE_MISSING"):
        _validate_task_graph(graph, _contract(), _map(), catalog)


def test_semantic_plan_rejects_stale_or_model_selected_detour() -> None:
    plan = SemanticPlan(
        ordered_targets=["stale-gate", "pickup", "office"],
        rationale_summary="Old destination then current mission.",
    )
    with pytest.raises(MissionPreparationBlocked, match="SEMANTIC_PLAN_UNAUTHORIZED_TARGET"):
        _validate_semantic_plan(plan, _contract(), _map())


def test_semantic_plan_accepts_contract_target_and_return_only() -> None:
    plan = SemanticPlan(
        ordered_targets=["pickup", "office"],
        rationale_summary="Current target and contract return.",
    )
    _validate_semantic_plan(plan, _contract(), _map())


def test_explicit_safety_priority_must_be_structured_not_only_goal_prose() -> None:
    request = MissionRequest(
        conversation_id="thread-1",
        message="恢复去外卖点取件并返回办公室，同时保持安全优先。",
    )
    hints = _explicit_constraint_hints(request)
    intent = IntentArtifact(
        goal="取件并返回，保持安全优先",
        start_entity="office",
        target_entity="pickup",
        return_entity="office",
        payload_action="pickup",
        constraints=[],
    )

    assert hints == ["safety_priority"]
    assert _missing_explicit_constraints(intent, hints) == ["safety_priority"]

    intent.constraints = ["safety_priority"]
    assert _missing_explicit_constraints(intent, hints) == []


def test_long_pickup_request_extracts_value_and_sensor_constraints() -> None:
    request = MissionRequest(
        conversation_id="thread-1",
        message=(
            "结合学校地图、前向视觉、深度感知、定位与飞行遥测飞到取餐点，"
            "连续稳定悬停10秒，等待挂载0.04公斤外卖，沿安全路径返回；"
            "先生成计划，不要开始执行。"
        ),
    )

    assert _explicit_constraint_hints(request) == [
        "plan_only",
        "do_not_execute",
        "safe_return_path",
        "requires_school_map",
        "requires_forward_vision",
        "requires_depth_sensing",
        "requires_localization",
        "requires_flight_telemetry",
        "pickup_hover_seconds=10",
        "payload_mass_kg=0.04",
    ]


def test_explicit_impossible_payload_is_rejected_before_model_planning() -> None:
    vehicle = VehicleAsset(
        asset_id="test-vehicle",
        name="Test vehicle",
        dry_mass_kg=2.0,
        max_takeoff_mass_kg=2.05,
        body_radius_m=0.3,
        body_height_m=0.4,
        max_speed_mps=1.0,
        max_acceleration_mps2=0.8,
        qualified_range_m=100.0,
        reserve_battery_percent=30.0,
        max_pickup_payload_kg=0.08,
        sensors=["depth"],
    )

    with pytest.raises(
        MissionPreparationBlocked,
        match="EXPLICIT_PAYLOAD_EXCEEDS_VEHICLE_CAPACITY",
    ):
        _validate_explicit_payload_capacity(["payload_mass_kg=0.06"], vehicle)

    _validate_explicit_payload_capacity(["payload_mass_kg=0.04"], vehicle)


def test_unresolved_critical_fields_produce_actionable_fail_closed_feedback() -> None:
    intent = IntentArtifact(
        goal="retrieve parcel and return",
        start_entity="office",
        target_entity="pickup",
        return_entity="office",
        payload_action="pickup",
        missing_critical_fields=["occupancy_layer", "esdf_layer"],
    )
    feedback = _intent_repair_feedback(
        intent,
        IntentCritique(accepted=True),
        [],
    )

    assert feedback["accepted"] is False
    assert feedback["issue_codes"] == ["UNRESOLVED_CRITICAL_FIELDS"]
    assert "occupancy_layer, esdf_layer" in feedback["repair_instructions"][0]
    assert "otherwise keep it" in feedback["repair_instructions"][0]


def test_critic_feedback_is_preserved_when_no_deterministic_gate_rejects() -> None:
    intent = IntentArtifact(
        goal="retrieve parcel and return",
        start_entity="office",
        target_entity="pickup",
        return_entity="office",
        payload_action="pickup",
    )
    critique = IntentCritique(
        accepted=False,
        issue_codes=["wrong-return"],
        repair_instructions=["Restore the requested return entity."],
    )

    assert _intent_repair_feedback(intent, critique, []) == critique.model_dump(mode="json")


def test_intent_critic_cannot_block_a_resolved_pickup_with_downstream_checks() -> None:
    request = MissionRequest(
        conversation_id="conversation-test",
        message="帮我取一下外卖。补充指令：外卖放在 L1 物资区，去那里取。",
        start_entity="launch",
    )
    intent = _bind_product_intent_defaults(
        IntentArtifact(
            goal="从 L1 物资区取外卖并返回",
            start_entity="launch",
            target_entity="l1-supplies",
            return_entity="launch",
            payload_action="pickup",
        ),
        request,
    )
    critique = IntentCritique(
        accepted=False,
        issue_codes=[
            "MISSING_EXPLICIT_CONSTRAINT_HINT_IN_CONSTRAINTS",
            "RETURN_ENTITY_ROUND_TRIP_MISMATCH",
            "ENVIRONMENT_MODE_UNDERSTATEMENT",
            "PICKUP_WAYPOINT_SEMANTIC_MISSING",
            "MISSING_PAYLOAD_MASS_CONSTRAINT",
        ],
        repair_instructions=["Run checks owned by later preparation stages."],
    )

    normalized, discarded = _normalize_intent_critique_scope(critique, intent, request)

    assert normalized == IntentCritique(accepted=True)
    assert discarded == critique.issue_codes


def test_intent_critic_keeps_a_real_target_semantic_mismatch() -> None:
    intent = IntentArtifact(
        goal="去物资区取外卖",
        start_entity="launch",
        target_entity="l1-supplies",
        return_entity="launch",
        payload_action="pickup",
    )
    critique = IntentCritique(
        accepted=False,
        issue_codes=["TARGET_ENTITY_MISMATCH", "MISSING_PAYLOAD_MASS_CONSTRAINT"],
        repair_instructions=["Use the place selected by the user."],
    )

    normalized, discarded = _normalize_intent_critique_scope(
        critique,
        intent,
        MissionRequest(
            conversation_id="conversation-test",
            message="去休息室取外卖",
            start_entity="launch",
        ),
    )

    assert normalized.accepted is False
    assert normalized.issue_codes == ["TARGET_ENTITY_MISMATCH"]
    assert discarded == ["MISSING_PAYLOAD_MASS_CONSTRAINT"]


def test_manifest_conditions_recommend_plugins_without_hardcoded_plugin_ids() -> None:
    catalog = [
        {
            "tool_id": "example.payload-audit",
            "routing_metadata": {"recommended_when": {"payload_action_in": ["pickup"]}},
        },
        {
            "tool_id": "example.unrelated",
            "routing_metadata": {"recommended_when": {"payload_action_in": ["none"]}},
        },
    ]

    assert _recommended_plugin_tools(catalog, _contract()) == ["example.payload-audit"]
