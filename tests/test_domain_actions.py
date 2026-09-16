from __future__ import annotations

import json

import pytest

from dronedream_agent_core.contracts import (
    MapAsset,
    MissionContract,
    RuntimeCheckpoint,
    RuntimeCheckpointContract,
    TaskGraph,
    VehicleAsset,
)
from dronedream_agent_core.domain_actions import (
    DomainActionError,
    action_by_id,
    action_ids,
    merge_action_packs,
    movement_action_ids,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.orchestrator import MissionPreparationBlocked, _validate_task_graph
from dronedream_agent_core.plugin_api import build_discovered_extension_registry
from dronedream_agent_core.runtime_actions import (
    build_runtime_action_execution_contract,
    merge_runtime_action_adapters,
)


def _catalog():
    registry = build_discovered_extension_registry()
    outputs, receipts = registry.invoke_multiple("mission.action-packs", "declare_actions")
    assert outputs
    assert all(receipt.outcome == "accepted" for receipt in receipts)
    return merge_action_packs(outputs)


def _map() -> MapAsset:
    return MapAsset.model_validate(
        {
            "asset_id": "school-map",
            "name": "School Map",
            "nodes": [
                {
                    "node_id": "office",
                    "label": "Office",
                    "position_m": {"x": 0, "y": 0, "z": 1},
                    "semantic": "launch",
                },
                {
                    "node_id": "pickup",
                    "label": "Pickup",
                    "position_m": {"x": 2, "y": 0, "z": 1},
                    "semantic": "pickup",
                },
            ],
            "edges": [
                {
                    "edge_id": "office-pickup",
                    "from_node": "office",
                    "to_node": "pickup",
                    "distance_m": 2,
                    "minimum_clearance_m": 1,
                    "speed_limit_mps": 1,
                }
            ],
            "named_entities": {"office": "office", "pickup": "pickup"},
        }
    )


def _contract(catalog) -> MissionContract:
    return MissionContract(
        contract_id="mission-0123456789abcdef01234567",
        conversation_id="thread-1",
        goal="retrieve parcel",
        start_node="office",
        target_node="pickup",
        return_node="office",
        payload_action="pickup",
        domain_ids=catalog.domain_ids,
        authorized_actions=sorted(action_ids(catalog)),
        action_catalog_sha256=sha256_json(catalog),
        map_asset_id="school-map",
        map_sha256="a" * 64,
        map_semantic_sha256="b" * 64,
        vehicle_asset_id="drone",
        vehicle_sha256="c" * 64,
        constraints=[],
        immutable_safety_rules=["model has no actuator authority"],
    )


def _vehicle(*, maximum_payload_kg: float = 0.1) -> VehicleAsset:
    return VehicleAsset(
        asset_id="drone",
        name="Test drone",
        dry_mass_kg=2.0,
        max_takeoff_mass_kg=2.2,
        body_radius_m=0.38,
        body_height_m=0.43,
        max_speed_mps=1.2,
        max_acceleration_mps2=0.8,
        qualified_range_m=400.0,
        reserve_battery_percent=30.0,
        max_pickup_payload_kg=maximum_payload_kg,
        sensors=["gps", "odometry"],
    )


def test_first_party_action_packs_merge_without_conflict() -> None:
    catalog = _catalog()

    assert {"core.flight", "delivery.custody", "inspection.infrastructure"}.issubset(
        set(catalog.domain_ids)
    )
    assert {
        "takeoff",
        "land",
        "pickup",
        "inspection.capture-thermal",
        "delivery.verify-release-area",
        "delivery.precontact-hold",
        "delivery.confirm-custody",
        "delivery.verify-loaded-stability",
        "emergency.verify-drop-zone",
    }.issubset(action_ids(catalog))
    assert "survey.grid-capture" in movement_action_ids(catalog)
    assert action_by_id(catalog, "delivery.release-payload").authority == "actuate"


def test_action_pack_conflict_is_rejected() -> None:
    base = {
        "schema_version": "dronedream.action-definition.v1",
        "action_id": "takeoff",
        "domain_id": "core.flight",
        "label": "Takeoff",
        "description": "Takeoff action",
        "movement": False,
        "payload": False,
        "flight_boundary": "takeoff",
        "input_schema": {},
        "required_success_evidence": ["airborne"],
        "allowed_fallbacks": ["land"],
        "simulator_executor": "sim.flight.takeoff",
        "runtime_executor": None,
        "authority": "plan",
    }
    with pytest.raises(DomainActionError, match="DOMAIN_ACTION_CONFLICT"):
        merge_action_packs(
            [
                {"domain_id": "core.flight", "actions": [base]},
                {
                    "domain_id": "core.flight",
                    "actions": [{**base, "description": "Different bytes"}],
                },
            ]
        )


def test_task_graph_enforces_registered_action_fallback_and_evidence() -> None:
    catalog = _catalog()
    contract = _contract(catalog)
    valid = TaskGraph.model_validate(
        {
            "nodes": [
                {
                    "task_id": "takeoff",
                    "action": "takeoff",
                    "target_node": "office",
                    "success_evidence": ["airborne", "stable hover"],
                    "fallback": "land",
                },
                {
                    "task_id": "navigate",
                    "action": "navigate",
                    "target_node": "pickup",
                    "depends_on": ["takeoff"],
                    "success_evidence": ["target reached", "pose stable"],
                    "fallback": "hold",
                },
                {
                    "task_id": "precontact",
                    "action": "delivery.precontact-hold",
                    "target_node": "pickup",
                    "depends_on": ["navigate"],
                    "success_evidence": ["stable pre-contact hover", "no payload contact"],
                    "fallback": "hold",
                },
                {
                    "task_id": "verify",
                    "action": "delivery.verify-recipient",
                    "target_node": "pickup",
                    "depends_on": ["precontact"],
                    "success_evidence": ["recipient identity accepted"],
                    "fallback": "hold",
                },
                {
                    "task_id": "pickup",
                    "action": "pickup",
                    "target_node": "pickup",
                    "depends_on": ["verify"],
                    "success_evidence": ["payload attached", "attachment state readback"],
                    "fallback": "return",
                },
                {
                    "task_id": "custody",
                    "action": "delivery.confirm-custody",
                    "target_node": "pickup",
                    "depends_on": ["pickup"],
                    "success_evidence": [
                        "payload attachment confirmed",
                        "mass and inertia update confirmed",
                        "custody state accepted",
                    ],
                    "fallback": "hold",
                },
                {
                    "task_id": "loaded-stability",
                    "action": "delivery.verify-loaded-stability",
                    "target_node": "pickup",
                    "depends_on": ["custody"],
                    "success_evidence": [
                        "loaded hover stable",
                        "post-attachment dynamics accepted",
                        "return authorized",
                    ],
                    "fallback": "hold",
                },
                {
                    "task_id": "return",
                    "action": "return",
                    "target_node": "office",
                    "depends_on": ["loaded-stability"],
                    "success_evidence": ["return node reached"],
                    "fallback": "land",
                },
                {
                    "task_id": "land",
                    "action": "land",
                    "target_node": "office",
                    "depends_on": ["return"],
                    "success_evidence": ["landed", "motors safe"],
                    "fallback": "hold",
                },
            ]
        }
    )
    _validate_task_graph(valid, contract, _map(), catalog)

    invalid = valid.model_copy(deep=True)
    invalid.nodes[4].fallback = "continue"
    with pytest.raises(MissionPreparationBlocked, match="TASK_GRAPH_ACTION_FALLBACK_UNAUTHORIZED"):
        _validate_task_graph(invalid, contract, _map(), catalog)


def test_runtime_action_contract_binds_verification_and_pickup_to_checkpoint(
    tmp_path,
) -> None:
    registry = build_discovered_extension_registry()
    action_outputs, _ = registry.invoke_multiple("mission.action-packs", "declare_actions")
    adapter_outputs, _ = registry.invoke_multiple(
        "runtime.action-adapters", "declare_runtime_action_adapters"
    )
    catalog = merge_action_packs(action_outputs)
    adapters = merge_runtime_action_adapters(adapter_outputs)
    contract = _contract(catalog)
    graph = TaskGraph.model_validate(
        {
            "nodes": [
                {
                    "task_id": "takeoff",
                    "action": "takeoff",
                    "target_node": "office",
                    "success_evidence": ["airborne", "stable hover"],
                    "fallback": "land",
                },
                {
                    "task_id": "navigate",
                    "action": "navigate",
                    "target_node": "pickup",
                    "depends_on": ["takeoff"],
                    "success_evidence": ["target reached", "pose stable"],
                    "fallback": "hold",
                },
                {
                    "task_id": "precontact",
                    "action": "delivery.precontact-hold",
                    "target_node": "pickup",
                    "depends_on": ["navigate"],
                    "success_evidence": ["stable pre-contact hover", "no payload contact"],
                    "fallback": "hold",
                },
                {
                    "task_id": "verify",
                    "action": "delivery.verify-recipient",
                    "target_node": "pickup",
                    "depends_on": ["precontact"],
                    "success_evidence": ["recipient identity accepted"],
                    "fallback": "hold",
                },
                {
                    "task_id": "pickup",
                    "action": "pickup",
                    "target_node": "pickup",
                    "depends_on": ["verify"],
                    "success_evidence": ["payload attached", "attachment state readback"],
                    "fallback": "return",
                },
                {
                    "task_id": "custody",
                    "action": "delivery.confirm-custody",
                    "target_node": "pickup",
                    "depends_on": ["pickup"],
                    "success_evidence": [
                        "payload attachment confirmed",
                        "mass and inertia update confirmed",
                        "custody state accepted",
                    ],
                    "fallback": "hold",
                },
                {
                    "task_id": "loaded-stability",
                    "action": "delivery.verify-loaded-stability",
                    "target_node": "pickup",
                    "depends_on": ["custody"],
                    "success_evidence": [
                        "loaded hover stable",
                        "post-attachment dynamics accepted",
                        "return authorized",
                    ],
                    "fallback": "hold",
                },
                {
                    "task_id": "return",
                    "action": "return",
                    "target_node": "office",
                    "depends_on": ["loaded-stability"],
                    "success_evidence": ["return node reached"],
                    "fallback": "land",
                },
                {
                    "task_id": "land",
                    "action": "land",
                    "target_node": "office",
                    "depends_on": ["return"],
                    "success_evidence": ["landed", "motors safe"],
                    "fallback": "hold",
                },
            ]
        }
    )
    checkpoints = RuntimeCheckpointContract(
        contract_id=contract.contract_id,
        checkpoints=[
            RuntimeCheckpoint(
                checkpoint_id="checkpoint-001",
                segment_id="segment-001",
                task_id="navigate",
                track_point_index=1,
                target_node="pickup",
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
    vehicle_sdf = tmp_path / "vehicle.sdf"
    vehicle_sdf.write_text(
        """<sdf version="1.10"><model name="drone"><plugin name="detachable_joint">
        <parent_link>base_link</parent_link>
        <child_model>payload</child_model>
        <child_link>payload_link</child_link>
        <attach_topic>/payload/attach</attach_topic>
        <detach_topic>/payload/detach</detach_topic>
        <output_topic>/payload/state</output_topic>
        </plugin></model></sdf>""",
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "model_name": "drone",
                "mission_payload": {
                    "model_name": "payload",
                    "center_above_model_root_m": 0.12,
                    "maximum_attachment_error_m": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "takeout-payload.sdf").write_text(
        """<sdf version="1.10"><model name="payload"><link name="payload_link">
        <inertial><mass>0.1</mass><inertia><ixx>0.01</ixx><iyy>0.02</iyy>
        <izz>0.03</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        </link></model></sdf>""",
        encoding="utf-8",
    )

    runtime = build_runtime_action_execution_contract(
        mission_contract=contract,
        task_graph=graph,
        domain_actions=catalog,
        adapter_catalog=adapters,
        checkpoints=checkpoints,
        vehicle=_vehicle(),
        vehicle_sdf=vehicle_sdf,
    )

    assert [step.task_id for step in runtime.steps] == [
        "precontact",
        "verify",
        "pickup",
        "custody",
        "loaded-stability",
    ]
    assert all(step.checkpoint_id == "checkpoint-001" for step in runtime.steps)
    assert runtime.steps[1].depends_on == ["precontact"]
    assert runtime.steps[2].depends_on == ["verify"]
    assert runtime.steps[3].depends_on == ["pickup"]
    assert runtime.steps[4].depends_on == ["custody"]
    assert runtime.steps[2].parameters["topic"] == "/payload/attach"
    assert runtime.steps[2].timeout_seconds == 25.0
    assert runtime.steps[2].max_attempts == 1
    assert runtime.steps[2].parameters["payload_mount_offset_model_m"] == [0.0, 0.0, 0.12]
    assert runtime.steps[0].parameters["operation"] == "precontact"
    assert runtime.steps[3].parameters["operation"] == "confirm-custody"
    assert runtime.steps[3].parameters["payload_mass_kg"] == pytest.approx(0.1)
    assert runtime.steps[4].parameters["operation"] == "postattach-stability"
    assert runtime.steps[1].parameters["request"]["contract_id"] == contract.contract_id
    assert runtime.steps[1].parameters["service_name"] == (
        "/dronedream/domain_actions/payload/verify_recipient"
    )

    with pytest.raises(
        RuntimeError, match="RUNTIME_ACTION_PAYLOAD_EXCEEDS_VEHICLE_CAPACITY"
    ):
        build_runtime_action_execution_contract(
            mission_contract=contract,
            task_graph=graph,
            domain_actions=catalog,
            adapter_catalog=adapters,
            checkpoints=checkpoints,
            vehicle=_vehicle(maximum_payload_kg=0.05),
            vehicle_sdf=vehicle_sdf,
        )
