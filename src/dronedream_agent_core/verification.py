"""Compile an explicit, hash-bound verification plan for an accepted mission.

The language model may propose tasks and evidence labels, but this module owns
the executable meaning of verification.  Every requirement names its lifecycle
phase, evidence authority and immutable source binding; model prose is never an
evidence authority.
"""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import (
    DomainActionCatalog,
    FlightPlan,
    GraphRoute,
    MissionContract,
    MissionVerificationPlan,
    MissionVerificationRequirement,
    Px4Track,
    RouteClearanceReport,
    RuntimeActionExecutionContract,
    RuntimeCheckpointContract,
    SemanticPlan,
    TaskGraph,
)
from .domain_actions import DomainActionError, action_by_id
from .hashing import sha256_json


class MissionVerificationPlanError(RuntimeError):
    """The accepted proposal cannot be mapped to complete verification evidence."""


# 功能：
#   忽略证据标签大小写及首尾空白，但不把模型改写、否定或近义解释当作相同证据。
# 输入：
#   left：任务或执行器声明的证据标签。
#   right：动作目录要求的证据标签。
# 输出：
#   matches：规范化后的标签集合是否相等。
def _same_evidence(left: Iterable[str], right: Iterable[str]) -> bool:
    matches = {item.casefold().strip() for item in left} == {
        item.casefold().strip() for item in right
    }
    return matches


# 功能：
#   1. 重新验证冻结输入，核对任务授权、段路径、检查点和动作执行合同的相互绑定。
#   2. 为确认前、执行中和结束时的义务指定真实证据来源；编译不代表已经完成飞行验收。
# 输入：
#   contract：用户意图对应的结构化任务合同和资产摘要。
#   domain_actions：本次授权的动作语义目录。
#   task_graph：包含依赖、动作和目标的任务图。
#   semantic_plan：模型提出的语义规划。
#   flight_plan：任务对应的具体路线段。
#   execution_route：已完成几何净空检查的整条路线。
#   route_clearance：绑定当前地图和路线的净空报告。
#   px4_track：执行器使用的飞控轨迹与坐标合同。
#   runtime_checkpoints：运行时到达及继续检查点。
#   runtime_actions：非移动动作的编译执行合同。
# 输出：
#   verification_plan：绑定精确输入摘要及证据权威的验证义务清单。
def build_mission_verification_plan(
    *,
    contract: MissionContract,
    domain_actions: DomainActionCatalog,
    task_graph: TaskGraph,
    semantic_plan: SemanticPlan,
    flight_plan: FlightPlan,
    execution_route: GraphRoute,
    route_clearance: RouteClearanceReport,
    px4_track: Px4Track,
    runtime_checkpoints: RuntimeCheckpointContract,
    runtime_actions: RuntimeActionExecutionContract,
) -> MissionVerificationPlan:
    # Pydantic 实例的嵌套列表可能已被改写；先重新校验副本，再计算摘要和义务。
    try:
        (
            contract,
            domain_actions,
            task_graph,
            semantic_plan,
            flight_plan,
            execution_route,
            route_clearance,
            px4_track,
            runtime_checkpoints,
            runtime_actions,
        ) = (
            model.model_validate(value.model_dump(mode="python"), strict=True)
            for model, value in zip(
                (
                    MissionContract,
                    DomainActionCatalog,
                    TaskGraph,
                    SemanticPlan,
                    FlightPlan,
                    GraphRoute,
                    RouteClearanceReport,
                    Px4Track,
                    RuntimeCheckpointContract,
                    RuntimeActionExecutionContract,
                ),
                (
                    contract,
                    domain_actions,
                    task_graph,
                    semantic_plan,
                    flight_plan,
                    execution_route,
                    route_clearance,
                    px4_track,
                    runtime_checkpoints,
                    runtime_actions,
                ),
                strict=True,
            )
        )
    except ValueError as error:
        raise MissionVerificationPlanError("MISSION_VERIFICATION_INPUT_INVALID") from error
    task_graph_sha256 = sha256_json(task_graph)
    semantic_plan_sha256 = sha256_json(semantic_plan)
    execution_route_sha256 = sha256_json(execution_route)
    tasks_by_id = {task.task_id: task for task in task_graph.nodes}
    try:
        definitions = {
            task.task_id: action_by_id(domain_actions, task.action) for task in task_graph.nodes
        }
    except DomainActionError as error:
        raise MissionVerificationPlanError("MISSION_VERIFICATION_UNKNOWN_TASK_ACTION") from error
    movement_task_ids = {
        task.task_id for task in task_graph.nodes if definitions[task.task_id].movement
    }
    runtime_action_task_ids = {
        task.task_id
        for task in task_graph.nodes
        if not definitions[task.task_id].movement
        and definitions[task.task_id].flight_boundary == "none"
        and task.action != "none"
    }
    segment_task_ids = {segment.task_id for segment in flight_plan.segments}
    checkpoint_task_ids = {checkpoint.task_id for checkpoint in runtime_checkpoints.checkpoints}
    segments_by_id = {segment.segment_id: segment for segment in flight_plan.segments}
    segment_offsets = {}
    flattened_path = []
    for index, segment in enumerate(flight_plan.segments):
        segment_offsets[segment.segment_id] = max(0, len(flattened_path) - 1)
        flattened_path.extend(segment.path if index == 0 else segment.path[1:])
    # 中间检查点允许绑定途经节点，但不能用途经节点替代最终到达任务目标的检查点。
    checkpoint_geometry_matches = True
    arrived_tasks = set()
    for checkpoint in runtime_checkpoints.checkpoints:
        segment = segments_by_id.get(checkpoint.segment_id)
        local_index = checkpoint.track_point_index - segment_offsets.get(checkpoint.segment_id, 0)
        if (
            segment is None
            or segment.task_id != checkpoint.task_id
            or not 0 <= local_index < len(segment.path)
            or not checkpoint.track_point_index < len(px4_track.points)
            or segment.path[local_index].node_id != checkpoint.target_node
        ):
            checkpoint_geometry_matches = False
            continue
        if (
            checkpoint.task_id in tasks_by_id
            and checkpoint.target_node == tasks_by_id[checkpoint.task_id].target_node
            and local_index == len(segment.path) - 1
        ):
            arrived_tasks.add(checkpoint.task_id)
    step_by_task = {step.task_id: step for step in runtime_actions.steps}
    task_evidence_matches_catalog = all(
        _same_evidence(
            task.success_evidence,
            definitions[task.task_id].required_success_evidence,
        )
        for task in task_graph.nodes
    )
    runtime_action_evidence_matches_catalog = all(
        step.task_id in tasks_by_id
        and step.action == tasks_by_id[step.task_id].action
        and _same_evidence(
            step.required_success_evidence,
            definitions[step.task_id].required_success_evidence,
        )
        for step in runtime_actions.steps
    )
    gates = {
        "action_catalog_bound_to_contract": sha256_json(domain_actions)
        == contract.action_catalog_sha256,
        "task_actions_authorized": all(
            task.action in contract.authorized_actions for task in task_graph.nodes
        ),
        "flight_plan_contract_matches": flight_plan.contract_id == contract.contract_id,
        "runtime_checkpoints_contract_matches": (
            runtime_checkpoints.contract_id == contract.contract_id
        ),
        "runtime_actions_contract_matches": runtime_actions.contract_id == contract.contract_id,
        "runtime_actions_task_graph_matches": (
            runtime_actions.task_graph_sha256 == task_graph_sha256
        ),
        "runtime_actions_domain_catalog_matches": (
            runtime_actions.domain_action_catalog_sha256 == sha256_json(domain_actions)
        ),
        "semantic_plan_bound_to_flight_plan": (
            flight_plan.semantic_plan_sha256 == semantic_plan_sha256
        ),
        "clearance_bound_to_execution_route": (
            route_clearance.route_sha256 == execution_route_sha256
        ),
        # Identical route coordinates are not enough: a different obstacle map
        # can make the same route unsafe, even with a genuine clearance receipt.
        "clearance_bound_to_mission_map": (
            route_clearance.semantic_sha256 == contract.map_semantic_sha256
        ),
        "continuous_route_clearance_accepted": route_clearance.accepted is True,
        "clearance_has_no_collision": route_clearance.collision_count == 0
        and not route_clearance.collisions,
        "route_starts_at_contract_start": (execution_route.start_node == contract.start_node),
        "route_ends_at_contract_return": (execution_route.goal_node == contract.return_node),
        "movement_tasks_have_flight_segments": movement_task_ids == segment_task_ids,
        "movement_tasks_have_runtime_checkpoints": movement_task_ids == checkpoint_task_ids,
        "movement_tasks_have_arrival_checkpoint": movement_task_ids <= arrived_tasks,
        "flight_segment_ids_unique": len(segments_by_id) == len(flight_plan.segments),
        "flight_segments_match_endpoints": all(
            segment.from_node == segment.path[0].node_id
            and segment.to_node == segment.path[-1].node_id
            for segment in flight_plan.segments
        ),
        "flight_segment_geometry_continuous": all(
            previous.path[-1] == current.path[0]
            for previous, current in zip(
                flight_plan.segments, flight_plan.segments[1:], strict=False
            )
        ),
        "flight_plan_bound_to_execution_route": (
            [point.node_id for point in flattened_path] == execution_route.node_ids
            and [point.position_m for point in flattened_path] == execution_route.positions_m
        ),
        "checkpoint_ids_unique": len({cp.checkpoint_id for cp in runtime_checkpoints.checkpoints})
        == len(runtime_checkpoints.checkpoints),
        "checkpoint_geometry_matches_flight_plan": checkpoint_geometry_matches,
        "runtime_action_tasks_match_steps": (runtime_action_task_ids == set(step_by_task)),
        # Check original list cardinality before its task-indexed view is used.
        # Otherwise a duplicate can disappear while still being executable.
        "runtime_action_steps_unique": (
            len(step_by_task)
            == len(runtime_actions.steps)
            == len({step.step_id for step in runtime_actions.steps})
        ),
        "runtime_action_steps_match_tasks": all(
            step.task_id in runtime_action_task_ids
            and step.action == tasks_by_id[step.task_id].action
            and step.target_node == tasks_by_id[step.task_id].target_node
            and step.fallback == tasks_by_id[step.task_id].fallback
            and step.runtime_executor == definitions[step.task_id].runtime_executor
            for step in runtime_actions.steps
        ),
        "task_evidence_matches_catalog": task_evidence_matches_catalog,
        "runtime_action_evidence_matches_catalog": (runtime_action_evidence_matches_catalog),
    }
    failed = [name for name, accepted in gates.items() if not accepted]
    if failed:
        raise MissionVerificationPlanError(
            "MISSION_VERIFICATION_PLAN_GATE_REJECTED:" + ",".join(failed)
        )

    requirements: list[MissionVerificationRequirement] = []

    # 功能：
    #   按稳定插入顺序建立验证义务，只指定要求和证据来源，不伪造已观测结果。
    # 输入：
    #   phase：义务生效的生命周期阶段。
    #   subject_type：被验证对象类别。
    #   subject_id：任务、路线、轨迹或动作标识。
    #   evidence_key：必须提供的精确证据标签。
    #   evidence_authority：有资格证明该事实的观测或执行组件。
    #   source_binding_sha256：义务所绑定输入的摘要。
    #   required_before：在什么事件之前必须验证通过。
    # 输出：
    #   None：向当前计划的 requirements 列表加入一项义务。
    def add(
        *,
        phase: str,
        subject_type: str,
        subject_id: str,
        evidence_key: str,
        evidence_authority: str,
        source_binding_sha256: str,
        required_before: str,
    ) -> None:
        requirements.append(
            MissionVerificationRequirement(
                requirement_id=f"verification-{len(requirements) + 1:03d}",
                phase=phase,  # type: ignore[arg-type]
                subject_type=subject_type,  # type: ignore[arg-type]
                subject_id=subject_id,
                evidence_key=evidence_key,
                evidence_authority=evidence_authority,  # type: ignore[arg-type]
                source_binding_sha256=source_binding_sha256,
                required_before=required_before,  # type: ignore[arg-type]
            )
        )

    for subject_type, subject_id, evidence_key, authority, source in (
        (
            "mission",
            contract.contract_id,
            "mission contract and asset bindings are frozen",
            "deterministic-core",
            contract,
        ),
        (
            "mission",
            contract.contract_id,
            "task evidence exactly matches the registered action catalog",
            "deterministic-core",
            task_graph,
        ),
        (
            "route",
            f"{execution_route.start_node}->{execution_route.goal_node}",
            "continuous vehicle-envelope clearance accepted",
            "qualified-tool",
            route_clearance,
        ),
        (
            "track",
            contract.contract_id,
            "PX4 track is derived from the selected hash-bound route",
            "qualified-tool",
            px4_track,
        ),
        (
            "mission",
            contract.contract_id,
            "runtime checkpoints and action adapters are fully bound",
            "deterministic-core",
            runtime_actions,
        ),
    ):
        add(
            phase="pre-confirmation",
            subject_type=subject_type,
            subject_id=subject_id,
            evidence_key=evidence_key,
            evidence_authority=authority,
            source_binding_sha256=sha256_json(source),
            required_before="user-confirmation",
        )

    segments_by_task = {
        task_id: [segment for segment in flight_plan.segments if segment.task_id == task_id]
        for task_id in movement_task_ids
    }
    for task in task_graph.nodes:
        definition = definitions[task.task_id]
        # Use the authority that can actually observe the fact. A planned route
        # proves neither arrival nor landing, and a model saying 'done' proves neither.
        if definition.flight_boundary == "landing":
            phase = "completion"
            authority = "px4-telemetry"
            required_before = "mission-completion"
            source_sha256 = sha256_json(task)
            subject_type = "landing"
        elif definition.flight_boundary == "takeoff":
            phase = "runtime"
            authority = "px4-telemetry"
            required_before = "task-advance"
            source_sha256 = sha256_json(task)
            subject_type = "task"
        elif definition.movement:
            phase = "runtime"
            authority = "runtime-checkpoint"
            required_before = "task-advance"
            source_sha256 = sha256_json(segments_by_task[task.task_id])
            subject_type = "task"
        elif task.action == "none":
            phase = "runtime"
            authority = "deterministic-core"
            required_before = "task-advance"
            source_sha256 = sha256_json(task)
            subject_type = "task"
        else:
            phase = "runtime"
            authority = "runtime-action-adapter"
            required_before = "task-advance"
            source_sha256 = sha256_json(step_by_task[task.task_id])
            subject_type = "runtime-action"
        for evidence_key in definition.required_success_evidence:
            add(
                phase=phase,
                subject_type=subject_type,
                subject_id=task.task_id,
                evidence_key=evidence_key,
                evidence_authority=authority,
                source_binding_sha256=source_sha256,
                required_before=required_before,
            )

    completion_binding = sha256_json(
        {
            "contract_id": contract.contract_id,
            "route_sha256": execution_route_sha256,
            "track_sha256": sha256_json(px4_track),
        }
    )
    for evidence_key, authority in (
        ("no live safety abort remains active", "deterministic-core"),
        ("all required runtime evidence gates are true", "deterministic-core"),
        ("completion assessment accepts the observed mission", "completion-verifier"),
    ):
        add(
            phase="completion",
            subject_type="mission",
            subject_id=contract.contract_id,
            evidence_key=evidence_key,
            evidence_authority=authority,
            source_binding_sha256=completion_binding,
            required_before="mission-completion",
        )

    verification_plan = MissionVerificationPlan(
        contract_id=contract.contract_id,
        plan_revision=flight_plan.revision,
        contract_sha256=sha256_json(contract),
        task_graph_sha256=task_graph_sha256,
        semantic_plan_sha256=semantic_plan_sha256,
        flight_plan_sha256=sha256_json(flight_plan),
        execution_route_sha256=execution_route_sha256,
        route_clearance_sha256=sha256_json(route_clearance),
        px4_track_sha256=sha256_json(px4_track),
        runtime_checkpoints_sha256=sha256_json(runtime_checkpoints),
        runtime_actions_sha256=sha256_json(runtime_actions),
        deterministic_gates=gates,
        requirements=requirements,
    )
    return verification_plan


# 功能：
#   重编译完整验证义务并比较，拒绝缺项、改写或绑定输入失效的既有计划。
# 输入：
#   verification_plan：需要核验的既有验证计划。
#   contract：当前任务合同。
#   domain_actions：当前动作目录。
#   task_graph：当前任务依赖图。
#   semantic_plan：当前语义规划。
#   flight_plan：当前飞行路线段。
#   execution_route：当前待执行路线。
#   route_clearance：该路线和地图的净空证据。
#   px4_track：当前飞控轨迹。
#   runtime_checkpoints：当前运行检查点。
#   runtime_actions：当前非移动动作执行合同。
# 输出：
#   matches：输入合法且全部义务与重编译结果完全一致时为真。
def verification_plan_matches_prepared_mission(
    *,
    verification_plan: MissionVerificationPlan,
    contract: MissionContract,
    domain_actions: DomainActionCatalog,
    task_graph: TaskGraph,
    semantic_plan: SemanticPlan,
    flight_plan: FlightPlan,
    execution_route: GraphRoute,
    route_clearance: RouteClearanceReport,
    px4_track: Px4Track,
    runtime_checkpoints: RuntimeCheckpointContract,
    runtime_actions: RuntimeActionExecutionContract,
) -> bool:
    try:
        verification_plan = MissionVerificationPlan.model_validate(
            verification_plan.model_dump(mode="python"),
            strict=True,
        )
        expected = build_mission_verification_plan(
            contract=contract,
            domain_actions=domain_actions,
            task_graph=task_graph,
            semantic_plan=semantic_plan,
            flight_plan=flight_plan,
            execution_route=execution_route,
            route_clearance=route_clearance,
            px4_track=px4_track,
            runtime_checkpoints=runtime_checkpoints,
            runtime_actions=runtime_actions,
        )
    except (MissionVerificationPlanError, ValueError):
        return False
    matches = verification_plan == expected
    return matches


__all__ = [
    "MissionVerificationPlanError",
    "build_mission_verification_plan",
    "verification_plan_matches_prepared_mission",
]
