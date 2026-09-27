"""Compile explicit simulation measurement tasks with production action contracts."""

from pathlib import Path

from ..contracts import (
    GraphRoute,
    MapAsset,
    MissionContract,
    RuntimeCheckpoint,
    RuntimeCheckpointContract,
    TaskGraph,
    TaskNode,
    VehicleAsset,
)
from ..domain_actions import action_by_id, merge_action_packs
from ..hashing import sha256_json
from ..orchestrator import _validate_task_graph
from ..plugin_api import build_discovered_extension_registry
from ..runtime_actions import build_runtime_action_execution_contract, merge_runtime_action_adapters


# 功能：
#   1. 将单次往返教师路线绑定到已知实验载荷的挂载、确认与稳定性检查，不声称收件人识别。
#   2. 使用当前插件目录、机型载荷资产及正式动作编译器，保留质量、惯量和依赖检查。
# 输入：
#   route、graph：同一次几何校验的完整往返路线及地图节点。
#   vehicle、vehicle_sdf：本次机型契约及原始物理资产。
#   semantic_sha256：本次碰撞地图的内容摘要。
# 输出：
#   artifacts：教师任务、检查点、运行时动作和目录，明确不授予模型飞行资格。
def build_payload_teacher_curriculum(
    route: GraphRoute, graph: MapAsset, vehicle: VehicleAsset,
    vehicle_sdf: Path, semantic_sha256: str,
):
    route = GraphRoute.model_validate(route.model_dump(), strict=True)
    graph = MapAsset.model_validate(graph.model_dump(), strict=True)
    vehicle = VehicleAsset.model_validate(vehicle.model_dump(), strict=True)
    count = len(route.node_ids)
    turn = count // 2
    if (count < 3 or count % 2 != 1 or route.start_node != route.goal_node
            or route.node_ids[:turn + 1] != list(reversed(route.node_ids[turn:]))
            or route.positions_m[:turn + 1] != list(reversed(route.positions_m[turn:]))
            or len(set(route.node_ids[:turn + 1])) != turn + 1):
        raise ValueError('PAYLOAD_TEACHER_REQUIRES_SINGLE_CLOSED_ROUTE')
    nodes = {node.node_id: node for node in graph.nodes}
    if (len(nodes) != len(graph.nodes) or any(
            identity not in nodes or nodes[identity].position_m != position
            for identity, position in zip(route.node_ids, route.positions_m, strict=True))
            or nodes[route.node_ids[turn]].semantic != 'pickup'):
        raise ValueError('PAYLOAD_TEACHER_ROUTE_GRAPH_MISMATCH')
    registry = build_discovered_extension_registry()
    actions, action_receipts = registry.invoke_multiple('mission.action-packs', 'declare_actions')
    adapters, adapter_receipts = registry.invoke_multiple(
        'runtime.action-adapters', 'declare_runtime_action_adapters')
    if not action_receipts or not adapter_receipts or any(
            item.outcome != 'accepted' for item in [*action_receipts, *adapter_receipts]):
        raise ValueError('PAYLOAD_TEACHER_PLUGIN_DECLARATION_REJECTED')
    catalog = merge_action_packs(actions)
    # 物理采集不包含“识别收件人”任务。专用动作只存在于本地采集目录，
    # 不注册到用户任务插件；实际挂载、质量和稳定性仍走原生物理驱动。
    fixture_attach = action_by_id(catalog, 'pickup').model_dump(mode='json')
    fixture_attach.update(action_id='training.payload-attach-fixture', domain_id='training.payload',
        label='Attach known simulation fixture',
        description='Physical measurement of the hash-bound simulation payload; no recipient claim')
    catalog = merge_action_packs([*actions, {'domain_id': 'training.payload',
                                           'actions': [fixture_attach]}])
    adapter_catalog = merge_runtime_action_adapters(adapters)
    start, target = route.start_node, route.node_ids[turn]
    specifications = [
        ('takeoff', 'takeoff', start, 'land'),
        ('navigate', 'navigate', target, 'hold'),
        ('precontact', 'delivery.precontact-hold', target, 'hold'),
        ('attach-fixture', 'training.payload-attach-fixture', target, 'return'),
        ('custody', 'delivery.confirm-custody', target, 'hold'),
        ('loaded-stability', 'delivery.verify-loaded-stability', target, 'hold'),
        ('return', 'return', start, 'land'),
        ('land', 'land', start, 'hold'),
    ]
    tasks = []
    for task_id, action, node, fallback in specifications:
        definition = action_by_id(catalog, action)
        tasks.append(TaskNode(task_id=task_id, action=action, target_node=node,
            depends_on=[tasks[-1].task_id] if tasks else [],
            success_evidence=list(definition.required_success_evidence), fallback=fallback))
    task_graph = TaskGraph(nodes=tasks)
    identity = sha256_json({'route': route, 'map': graph, 'vehicle': vehicle,
                           'semantic': semantic_sha256, 'tasks': task_graph})
    mission = MissionContract(contract_id='mission-' + identity[:24],
        conversation_id='simulation-payload-teacher-' + identity[:24],
        goal='Explicit simulation payload measurement curriculum; not a user/model mission',
        start_node=start, target_node=target, return_node=start,
        payload_action='training.payload-attach-fixture',
        domain_ids=catalog.domain_ids, authorized_actions=sorted({task.action for task in tasks}),
        action_catalog_sha256=sha256_json(catalog), map_asset_id=graph.asset_id,
        map_sha256=sha256_json(graph), map_semantic_sha256=semantic_sha256,
        vehicle_asset_id=vehicle.asset_id, vehicle_sha256=sha256_json(vehicle),
        constraints=['simulation-only', 'measurement-only', 'no model approval'],
        immutable_safety_rules=['Independent live safety must remain active',
                                'No flight qualification or production model authority'])
    _validate_task_graph(task_graph, mission, graph, catalog)
    checkpoints = RuntimeCheckpointContract(contract_id=mission.contract_id, checkpoints=[
        RuntimeCheckpoint(checkpoint_id='checkpoint-001', segment_id='segment-001',
            task_id='navigate', track_point_index=turn, target_node=target),
        RuntimeCheckpoint(checkpoint_id='checkpoint-002', segment_id='segment-002',
            task_id='return', track_point_index=count - 1, target_node=start)])
    execution = build_runtime_action_execution_contract(mission_contract=mission,
        task_graph=task_graph, domain_actions=catalog, adapter_catalog=adapter_catalog,
        checkpoints=checkpoints, vehicle=vehicle, vehicle_sdf=vehicle_sdf)
    artifacts = {'mission': mission, 'task_graph': task_graph, 'checkpoints': checkpoints,
                 'actions': execution, 'action_catalog': catalog,
                 'adapter_catalog': adapter_catalog}
    return artifacts
