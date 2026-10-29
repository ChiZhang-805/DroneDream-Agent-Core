"""Physical asset qualification, separate from model approval and teacher data."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from .contracts import (
    GraphRoute,
    MapAsset,
    MissionContract,
    RuntimeActionExecutionContract,
    RuntimeActionExecutionReceipt,
    RuntimeAssessment,
    RuntimeCheckpoint,
    RuntimeCheckpointContract,
    RuntimeCheckpointRequest,
    StrictModel,
    TaskGraph,
    TaskNode,
    Vector3,
    VehicleAsset,
)
from .domain_actions import action_by_id, merge_action_packs
from .hashing import sha256_json
from .plugin_api import build_discovered_extension_registry
from .runtime_actions import build_runtime_action_execution_contract, merge_runtime_action_adapters
from .simulation_payload_runtime import LIBRARY as PLACEMENT_LIBRARY

PAYLOAD_ACTIONS = (
    "delivery.precontact-hold",
    "pickup",
    "delivery.confirm-custody",
    "delivery.verify-loaded-stability",
    "delivery.confirm-custody",
    "delivery.verify-loaded-stability",
    "delivery.verify-release-area",
    "delivery.release-payload",
)
PAYLOAD_GATES = (
    "qualification_action_receipts_verified",
    "qualification_checkpoints_verified",
    "physical_payload_spawned",
    "physical_payload_attached",
    "physical_payload_retained",
    "physical_payload_released",
    "loaded_thrust_to_weight_qualified",
    "return_battery_reserve_preserved",
)
NATIVE_GATES = (
    "executor_completed",
    "offboard_timing_complete",
    "runtime_pose_samples_present",
    "ros_observations_present",
    "goal_observed",
    "landing_confirmed",
    "native_terminal_lifecycle_published",
    "no_live_abort",
    "px4_ulog_present",
    "static_route_clearance_bound",
    "development_fault_injection_absent",
    "development_payload_collection_absent",
    "offline_training_authority_absent",
    "native_sensor_subscriptions_closed",
    "live_depth_perception_healthy",
    "live_depth_metric_map_present",
    "live_depth_safety_history_present",
    "controlled_vehicle_entity_selected",
    "controlled_vehicle_identity_confirmed",
)


class AssetCheckpointDecision(StrictModel):
    schema_version: Literal["dronedream.asset-qualification-checkpoint.v1"] = (
        "dronedream.asset-qualification-checkpoint.v1"
    )
    request_sha256: str
    checkpoint_contract_sha256: str
    assessment: RuntimeAssessment
    continue_authorized: bool
    model_call_performed: Literal[False] = False
    qualification_scope: Literal["physical-assets-only"] = "physical-assets-only"


# 功能：核对当前原生遥测与冻结检查点；该回执只用于物理资产资格，不代签模型决定。
# 输入：request 为执行器刚取得的稳定到达证据，contract 为本轮冻结合同。
# 输出：单独类型的接受或拒绝回执。
def decide_asset_checkpoint(request, contract):
    request = RuntimeCheckpointRequest.model_validate(request.model_dump(), strict=True)
    contract = RuntimeCheckpointContract.model_validate(contract.model_dump(), strict=True)
    matches = [cp for cp in contract.checkpoints if cp == request.checkpoint]
    if request.contract_id != contract.contract_id or len(matches) != 1:
        raise ValueError("ASSET_CHECKPOINT_BINDING_INVALID")
    position, target = request.observed_position_ned_m, request.commanded_position_ned_m
    error = math.dist((position.x, position.y, position.z), (target.x, target.y, target.z))
    velocity = request.observed_velocity_ned_mps
    speed = math.hypot(velocity.x, velocity.y, velocity.z)
    if not math.isclose(error, request.position_error_m, abs_tol=1e-6) or not math.isclose(
        speed, request.speed_mps, abs_tol=1e-6
    ):
        raise ValueError("ASSET_CHECKPOINT_TELEMETRY_MISMATCH")
    gates = {
        "position_error_within_0_75_m": error <= 0.75,
        "speed_within_0_50_mps": speed <= 0.5,
        "battery_above_10_percent": request.battery_percent > 10.0,
        "telemetry_finite": True,
    }
    if gates != request.deterministic_gates:
        raise ValueError("ASSET_CHECKPOINT_GATES_MISMATCH")
    accepted = all(gates.values())
    return AssetCheckpointDecision(
        request_sha256=sha256_json(request),
        checkpoint_contract_sha256=sha256_json(contract),
        continue_authorized=accepted,
        assessment=RuntimeAssessment(
            action="accept" if accepted else "abort",
            issue_codes=[] if accepted else ["ASSET_CHECKPOINT_REJECTED"],
        ),
    )


# 功能：在原返程路线上放置独立卸货点，使落地足迹与已释放的包裹分离，不生成捷径。
# 输入：单次往返路线、检查图、机型；最后原路段必须足够长以容纳至少两米分离。
# 输出：只细分原有路段的路线和图，随后仍须通过完整净空核验。
def separate_release_from_landing(route: GraphRoute, graph: MapAsset, vehicle: VehicleAsset):
    route, graph = route.model_copy(deep=True), graph.model_copy(deep=True)
    if route.start_node != route.goal_node or route.positions_m[0] != route.positions_m[-1]:
        raise ValueError("ASSET_PAYLOAD_REQUIRES_RETURN_TO_LAUNCH")
    distance = max(2.0, 4 * vehicle.body_radius_m + 0.1)
    landing = route.positions_m[-1]
    turn = len(route.positions_m) // 2
    for index in range(len(route.positions_m) - 2, turn - 1, -1):
        first, last = route.positions_m[index : index + 2]
        length = math.dist((first.x, first.y, first.z), (last.x, last.y, last.z))
        if math.hypot(first.x - landing.x, first.y - landing.y) < distance:
            continue
        # Separation is a horizontal footprint, not accumulated 3D path length.
        # Intersect the original return segment with that footprint's boundary;
        # slopes and bends must not place the parcel back inside the landing area.
        dx, dy = first.x - last.x, first.y - last.y
        lx, ly = last.x - landing.x, last.y - landing.y
        a = dx * dx + dy * dy
        b = 2 * (lx * dx + ly * dy)
        c = lx * lx + ly * ly - distance * distance
        from_last = (-b + math.sqrt(max(0.0, b * b - 4 * a * c))) / (2 * a)
        fraction = 1 - from_last
        remaining = from_last * length
        if fraction <= 1e-8:
            release_id = route.node_ids[index]
            graph.named_entities["qualification-release"] = release_id
            return route, graph
        release_id = "qualification-release"
        if release_id in route.node_ids:
            raise ValueError("ASSET_PAYLOAD_RELEASE_ID_ALREADY_USED")
        point = Vector3(
            **{
                axis: getattr(first, axis) + fraction * (getattr(last, axis) - getattr(first, axis))
                for axis in ("x", "y", "z")
            }
        )
        original_id = route.edge_ids[index]
        edge = next(item for item in graph.edges if item.edge_id == original_id)
        first_id, last_id = original_id + "-unload", original_id + "-land"
        graph.edges = [item for item in graph.edges if item.edge_id != original_id]
        graph.edges.extend(
            [
                edge.model_copy(
                    update={
                        "edge_id": first_id,
                        "to_node": release_id,
                        "distance_m": length - remaining,
                    }
                ),
                edge.model_copy(
                    update={"edge_id": last_id, "from_node": release_id, "distance_m": remaining}
                ),
            ]
        )
        graph.nodes.append(
            graph.nodes[0].model_copy(
                update={
                    "node_id": release_id,
                    "label": "Return unloading point",
                    "semantic": "corridor",
                    "position_m": point,
                }
            )
        )
        graph.named_entities["qualification-release"] = release_id
        route.node_ids.insert(index + 1, release_id)
        route.positions_m.insert(index + 1, point)
        route.edge_ids[index : index + 1] = [first_id, last_id]
        # Keep the same length: the inserted point lies on the original segment.
        return GraphRoute.model_validate(route.model_dump()), MapAsset.model_validate(
            graph.model_dump()
        )
    raise ValueError("ASSET_PAYLOAD_RETURN_TOO_SHORT_FOR_SEPARATE_RELEASE")


# 功能：为实际指定的单次往返编译正式设备动作，包含返航后的载荷保持及安全释放。
# 输入：本轮完整路线、检查图、机型及真实载荷资产；不假定任何建筑名称。
# 输出：可冻结的任务、检查点和生产适配器动作；本函数不授予资格。
def build_asset_payload_course(
    route: GraphRoute,
    graph: MapAsset,
    vehicle: VehicleAsset,
    vehicle_sdf: Path,
    semantic_sha256: str,
):
    route = GraphRoute.model_validate(route.model_dump(), strict=True)
    graph = MapAsset.model_validate(graph.model_dump(), strict=True)
    vehicle = VehicleAsset.model_validate(vehicle.model_dump(), strict=True)
    count = len(route.node_ids)
    nodes = {node.node_id: node for node in graph.nodes}
    pickups = [
        i
        for i, identity in enumerate(route.node_ids)
        if identity in nodes and nodes[identity].semantic == "pickup"
    ]
    release = graph.named_entities.get("qualification-release")
    turn = pickups[0] if len(pickups) == 1 else -1
    releases = [i for i in range(turn + 1, count - 1) if route.node_ids[i] == release]
    if (
        count < 4
        or len(pickups) != 1
        or len(releases) != 1
        or route.start_node != route.goal_node
        or route.positions_m[0] != route.positions_m[-1]
        or len(nodes) != len(graph.nodes)
        or any(
            identity not in nodes or nodes[identity].position_m != point
            for identity, point in zip(route.node_ids, route.positions_m, strict=True)
        )
        or nodes[route.node_ids[turn]].semantic != "pickup"
    ):
        raise ValueError("ASSET_PAYLOAD_REQUIRES_BOUND_SINGLE_ROUND_TRIP")
    release_point, landing_point = route.positions_m[releases[0]], route.positions_m[-1]
    if math.hypot(release_point.x - landing_point.x, release_point.y - landing_point.y) < max(
        1.5, 4 * vehicle.body_radius_m
    ):
        raise ValueError("ASSET_PAYLOAD_RELEASE_OVERLAPS_LANDING_FOOTPRINT")
    registry = build_discovered_extension_registry()
    packs, receipts = registry.invoke_multiple("mission.action-packs", "declare_actions")
    adapters, adapter_receipts = registry.invoke_multiple(
        "runtime.action-adapters", "declare_runtime_action_adapters"
    )
    if (
        not receipts
        or not adapter_receipts
        or any(receipt.outcome != "accepted" for receipt in [*receipts, *adapter_receipts])
    ):
        raise ValueError("ASSET_PAYLOAD_PLUGIN_DECLARATION_REJECTED")
    catalog, adapter_catalog = merge_action_packs(packs), merge_runtime_action_adapters(adapters)
    start, target = route.start_node, route.node_ids[turn]
    specifications = [
        ("takeoff", "takeoff", start),
        ("navigate", "navigate", target),
        ("precontact", PAYLOAD_ACTIONS[0], target),
        ("pickup", PAYLOAD_ACTIONS[1], target),
        ("custody", PAYLOAD_ACTIONS[2], target),
        ("loaded-stability", PAYLOAD_ACTIONS[3], target),
        ("return", "return", release),
        ("return-custody", PAYLOAD_ACTIONS[4], release),
        ("return-stability", PAYLOAD_ACTIONS[5], release),
        ("release-area", PAYLOAD_ACTIONS[6], release),
        ("release", PAYLOAD_ACTIONS[7], release),
        ("return-pad", "return", start),
        ("land", "land", start),
    ]
    tasks = []
    for task_id, action, node in specifications:
        definition = action_by_id(catalog, action)
        tasks.append(
            TaskNode(
                task_id=task_id,
                action=action,
                target_node=node,
                depends_on=[tasks[-1].task_id] if tasks else [],
                success_evidence=definition.required_success_evidence,
                fallback="land",
            )
        )
    task_graph = TaskGraph(nodes=tasks)
    digest = sha256_json(
        {
            "route": route,
            "graph": graph,
            "vehicle": vehicle,
            "semantic": semantic_sha256,
            "tasks": task_graph,
        }
    )
    # This is an explicit physical acceptance course, not a model-generated user
    # plan. Its fixed sequence additionally authorizes unloading at the return pad.
    mission = MissionContract(
        contract_id="mission-" + digest[:24],
        conversation_id="asset-qualification-" + digest[:24],
        goal="Qualify physical pickup, loaded return, safe release and landing",
        start_node=start,
        target_node=target,
        return_node=start,
        payload_action="pickup",
        domain_ids=catalog.domain_ids,
        authorized_actions=sorted({t.action for t in tasks}),
        action_catalog_sha256=sha256_json(catalog),
        map_asset_id=graph.asset_id,
        map_sha256=sha256_json(graph),
        map_semantic_sha256=semantic_sha256,
        vehicle_asset_id=vehicle.asset_id,
        vehicle_sha256=sha256_json(vehicle),
        constraints=["physical asset qualification; no model qualification"],
        immutable_safety_rules=[
            "Live native depth safety remains active",
            "Release only after return custody, stability and area checks",
        ],
    )
    checkpoints = RuntimeCheckpointContract(
        contract_id=mission.contract_id,
        checkpoints=[
            RuntimeCheckpoint(
                checkpoint_id="checkpoint-001",
                segment_id="segment-001",
                task_id="navigate",
                track_point_index=turn,
                target_node=target,
            ),
            RuntimeCheckpoint(
                checkpoint_id="checkpoint-002",
                segment_id="segment-002",
                task_id="return",
                track_point_index=releases[0],
                target_node=release,
            ),
            RuntimeCheckpoint(
                checkpoint_id="checkpoint-003",
                segment_id="segment-003",
                task_id="return-pad",
                track_point_index=count - 1,
                target_node=start,
            ),
        ],
    )
    actions = build_runtime_action_execution_contract(
        mission_contract=mission,
        task_graph=task_graph,
        domain_actions=catalog,
        adapter_catalog=adapter_catalog,
        checkpoints=checkpoints,
        vehicle=vehicle,
        vehicle_sdf=vehicle_sdf,
    )
    return {
        "mission": mission,
        "task_graph": task_graph,
        "checkpoints": checkpoints,
        "actions": actions,
        "action_catalog": catalog,
        "adapter_catalog": adapter_catalog,
    }


def _read(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("ASSET_QUALIFICATION_EVIDENCE_NOT_OBJECT")
    return value


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 功能：只消费真实设备回读、连续动作时序和检查点，重新生成可发布的物理资格事实。
# 输入：已结束的本次仿真目录，含原始 generic 运行证据和冻结输入。
# 输出：资格事实及所消费证据的摘要；缺件、错绑或任何失败都不能产生成功。
def verify_asset_payload_run(root: Path):
    source = _read(root / "mission_evidence.json")
    inputs = _read(root / "asset-qualification-inputs.json")
    if (
        source.get("schema_version") != "dronedream.generic-px4-gazebo-run.v1"
        or inputs.get("schema_version") != "dronedream.asset-qualification-inputs.v1"
        or source.get("status") != "verified"
        or not source.get("gates")
        or not all(v is True for v in source["gates"].values())
        or any(source["gates"].get(name) is not True for name in NATIVE_GATES)
    ):
        raise ValueError("ASSET_QUALIFICATION_RUNTIME_NOT_VERIFIED")
    files = {
        "mission_evidence.json": file_sha256(root / "mission_evidence.json"),
        "asset-qualification-inputs.json": file_sha256(root / "asset-qualification-inputs.json"),
    }
    for name, digest in inputs["run_files"].items():
        if (
            not (root / name).resolve().is_relative_to(root.resolve())
            or file_sha256(root / name) != digest
        ):
            raise ValueError("ASSET_QUALIFICATION_INPUT_CHANGED")
        files[name] = digest
    for field, name in [
        ("route_sha256", "mission-route.json"),
        ("track_sha256", "reference_track.json"),
        ("clearance_sha256", "clearance.json"),
    ]:
        if source["artifacts"].get(field) != files.get(name):
            raise ValueError("ASSET_QUALIFICATION_RUNTIME_INPUT_MISMATCH")
    for field in ("world_sha256", "semantic_sha256", "vehicle_sha256", "controller_params_sha256"):
        if source["artifacts"].get(field) != inputs["runtime_inputs"].get(field):
            raise ValueError("ASSET_QUALIFICATION_RUNTIME_INPUT_MISMATCH")
    actions = RuntimeActionExecutionContract.model_validate(_read(root / "runtime-actions.json"))
    checkpoints = RuntimeCheckpointContract.model_validate(_read(root / "runtime-checkpoints.json"))
    route = GraphRoute.model_validate(_read(root / "mission-route.json"))
    if (
        actions.contract_id != checkpoints.contract_id
        or tuple(step.action for step in actions.steps) != PAYLOAD_ACTIONS
        or len(checkpoints.checkpoints) != 3
    ):
        raise ValueError("ASSET_QUALIFICATION_COURSE_MISMATCH")
    pickup = route.node_ids.index(actions.steps[0].target_node)
    release_index = route.node_ids.index(actions.steps[4].target_node, pickup + 1)
    expected_indices = [pickup, release_index, len(route.node_ids) - 1]
    requests = []
    for cp, index in zip(checkpoints.checkpoints, expected_indices, strict=True):
        if cp.track_point_index != index or cp.target_node != route.node_ids[index]:
            raise ValueError("ASSET_QUALIFICATION_CHECKPOINT_ROUTE_MISMATCH")
        req_name = f"checkpoints/{cp.checkpoint_id}.request.json"
        decision_name = f"checkpoints/{cp.checkpoint_id}.asset-decision.json"
        request = RuntimeCheckpointRequest.model_validate(_read(root / req_name))
        decision = AssetCheckpointDecision.model_validate(_read(root / decision_name))
        if (
            request.checkpoint != cp
            or decision != decide_asset_checkpoint(request, checkpoints)
            or not decision.continue_authorized
        ):
            raise ValueError("ASSET_QUALIFICATION_CHECKPOINT_NOT_VERIFIED")
        requests.append(request)
        for name in (req_name, decision_name):
            files[name] = file_sha256(root / name)
    receipts, completed, prior_time = [], set(), None
    for step in actions.steps:
        name = f"runtime-actions/receipts/{step.step_id}.receipt.json"
        receipt = RuntimeActionExecutionReceipt.model_validate(_read(root / name))
        if (
            receipt.execution_contract_sha256 != sha256_json(actions)
            or receipt.step_sha256 != sha256_json(step)
            or any(
                getattr(receipt, key) != getattr(step, key)
                for key in ("step_id", "task_id", "action", "adapter_id", "runtime_executor")
            )
            or receipt.status != "accepted"
            or receipt.issue_codes
            or not all(receipt.deterministic_gates.values())
            or receipt.output.get("confirmed") is not True
            or not set(step.required_success_evidence) <= set(receipt.observed_success_evidence)
            or not set(step.depends_on) <= completed
            or receipt.attempts > step.max_attempts
            or receipt.started_at.utcoffset() is None
            or receipt.completed_at.utcoffset() is None
            or receipt.started_at > receipt.completed_at
            or (prior_time is not None and receipt.started_at < prior_time)
        ):
            raise ValueError("ASSET_QUALIFICATION_ACTION_NOT_VERIFIED")
        receipts.append(receipt)
        completed.add(step.task_id)
        prior_time = receipt.completed_at
        files[name] = file_sha256(root / name)
    spawn = _read(root / "payload_spawn.json")
    attach, custody, stable, returned, return_stable, area, release = [
        r.output for r in receipts[1:]
    ]
    alignment = attach.get("payload_mount_alignment", {})
    readback = alignment.get("attachment_pose_readback", {})
    positions = [
        readback.get(key)
        for key in ("payload_position_world_enu_m", "expected_position_world_enu_m")
    ]
    physical_alignment = bool(
        all(isinstance(p, list) and len(p) == 3 for p in positions)
        and readback.get("accepted") is True
        and math.isclose(math.dist(*positions), readback.get("alignment_error_m", -1), abs_tol=1e-8)
        and 0
        <= readback.get("alignment_error_m", -1)
        <= actions.steps[1].parameters["payload_mount_max_alignment_error_m"]
        and alignment.get("binding_sha256")
        == actions.steps[1].parameters["payload_mount_binding_sha256"]
    )
    for offset, output in ((2, custody), (4, returned)):
        if any(
            output.get(key) != actions.steps[offset].parameters.get(key)
            for key in ("payload_mass_kg", "payload_inertia_kg_m2", "payload_sdf_sha256")
        ):
            raise ValueError("ASSET_QUALIFICATION_PAYLOAD_PHYSICS_MISMATCH")
        if (
            output.get("custody_state_accepted") is not True
            or output.get("payload_physics_binding_confirmed") is not True
        ):
            raise ValueError("ASSET_QUALIFICATION_CUSTODY_NOT_CONFIRMED")
    for offset, output in ((3, stable), (5, return_stable)):
        if (
            output.get("loaded_hover_stable") is not True
            or output.get("return_authorized") is not True
            or any(
                not isinstance(output.get(field), (int, float))
                or not 0 <= output[field] <= actions.steps[offset].parameters[bound]
                for field, bound in (
                    ("position_error_m", "position_tolerance_m"),
                    ("speed_mps", "speed_tolerance_mps"),
                )
            )
        ):
            raise ValueError("ASSET_QUALIFICATION_LOADED_STABILITY_NOT_CONFIRMED")
    gates = {
        "qualification_action_receipts_verified": True,
        "qualification_checkpoints_verified": True,
        "physical_payload_spawned": (
            spawn.get("accepted") is True
            and spawn.get("checkpoint_id") == checkpoints.checkpoints[0].checkpoint_id
            and spawn.get("track_point_index") == expected_indices[0]
        ),
        "physical_payload_attached": (
            physical_alignment
            and attach.get("detached") is False
            and attach.get("transport") == "gazebo-transport"
            and attach.get("state_readback_source") in (
                "gazebo-detachable-joint-event", "gazebo-ecm-joint-query"
            )
        ),
        "physical_payload_retained": all(
            output.get("detached") is False for output in (custody, stable, returned, return_stable)
        ),
        "physical_payload_released": (
            release.get("detached") is True
            and release.get("operation") == "detach"
            and release.get("transport") == "gazebo-transport"
            and area.get("operation") == "verify-release-area"
            and area.get("minimum_predicted_clearance_m", -math.inf)
            >= actions.steps[6].parameters["minimum_clearance_m"]
        ),
        "loaded_thrust_to_weight_qualified": inputs["loaded_thrust_to_weight"]
        >= inputs["minimum_thrust_to_weight"],
        "return_battery_reserve_preserved": requests[-1].battery_percent
        >= inputs["reserve_battery_percent"],
    }
    if not all(gates.values()):
        raise ValueError(
            "ASSET_QUALIFICATION_PHYSICAL_GATE_FAILED:"
            + ",".join(k for k, v in gates.items() if not v)
        )
    files["payload_spawn.json"] = file_sha256(root / "payload_spawn.json")
    deployment = _read(root / "payload-placement/deployment.json")
    if (
        spawn.get("placement_runtime") != deployment
        or deployment.get("original_sha256") != actions.steps[2].parameters["payload_sdf_sha256"]
        or spawn.get("sdf_sha256") != file_sha256(root / "payload-placement/payload.sdf")
        or deployment.get("staged_sha256") != spawn.get("sdf_sha256")
        or deployment.get("library_sha256")
        != file_sha256(root / "payload-placement" / PLACEMENT_LIBRARY)
    ):
        raise ValueError("ASSET_QUALIFICATION_PLACEMENT_BINDING_INVALID")
    for name in ("payload-placement/deployment.json", "offboard_timing.json"):
        files[name] = file_sha256(root / name)
    # These temporary native resources are verified at the actual run, but are
    # not redistributed as executable content inside map/vehicle packages.
    native_files = {
        name: file_sha256(root / name)
        for name in ("payload-placement/payload.sdf", "payload-placement/" + PLACEMENT_LIBRARY)
    }
    return {
        "schema_version": "dronedream.asset-payload-qualification.v1",
        "status": "verified",
        "qualification_scope": "physical-assets-only",
        "model_qualification_granted": False,
        "gates": {**source["gates"], **gates},
        "artifacts": source["artifacts"],
        "measurements": {
            **source["measurements"],
            "qualified_route_length_m": route.route_length_m,
            "payload_mass_kg": custody["payload_mass_kg"],
        },
        "verified_source_files_sha256": files,
        "verified_native_runtime_files_sha256": native_files,
    }
