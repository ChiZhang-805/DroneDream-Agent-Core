"""Physical payload curricula retain production action and asset validation."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from dronedream_agent_core.contracts import GraphRoute, Vector3, VehicleAsset
from dronedream_agent_core.gazebo_adapter import _validate_execution_contract_binding
from dronedream_agent_core.training.payload_curriculum import build_payload_teacher_curriculum
from scripts import px4_checkpoint_executor as executor
from scripts.run_school_map_depth_qualification import _qualification_route


# 功能：
#   创建独立测试路线、真实可解析载荷资产和机型能力，不启动飞控或伪造运行回执。
# 输入：
#   tmp_path：测试隔离目录。
# 输出：
#   inputs：可交给教师课程编译器的路线、地图、机型和资产路径。
@pytest.fixture
def curriculum_inputs(tmp_path):
    outbound = GraphRoute(start_node='a', goal_node='b', node_ids=['a', 'b'], edge_ids=['ab'],
        positions_m=[Vector3(x=0., y=0., z=2.), Vector3(x=2., y=0., z=2.)],
        route_length_m=2., all_edges_flight_verified=False)
    route, graph = _qualification_route(outbound, point_count=None, speed_limit_mps=.2)
    vehicle = VehicleAsset(asset_id='fixture', name='fixture', dry_mass_kg=2.,
        max_takeoff_mass_kg=2.2, body_radius_m=.38, body_height_m=.43, max_speed_mps=1.2,
        max_acceleration_mps2=.8, qualified_range_m=400., reserve_battery_percent=30.,
        max_pickup_payload_kg=.1, sensors=['imu', 'odometry'])
    sdf = tmp_path / 'vehicle.sdf'
    sdf.write_text('''<sdf version="1.10"><model name="drone"><plugin name="detachable_joint">
        <parent_link>base_link</parent_link><child_model>payload</child_model>
        <child_link>payload_link</child_link><attach_topic>/payload/attach</attach_topic>
        <detach_topic>/payload/detach</detach_topic><output_topic>/payload/state</output_topic>
        </plugin></model></sdf>''', encoding='utf-8')
    (tmp_path / 'summary.json').write_text(json.dumps({'model_name': 'drone',
        'mission_payload': {'model_name': 'payload', 'center_above_model_root_m': .12,
                            'maximum_attachment_error_m': .02}}), encoding='utf-8')
    (tmp_path / 'takeout-payload.sdf').write_text('''<sdf version="1.10"><model name="payload">
        <link name="payload_link"><inertial><mass>0.1</mass><inertia><ixx>0.01</ixx>
        <iyy>0.02</iyy><izz>0.03</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia></inertial></link></model></sdf>''', encoding='utf-8')
    inputs = (route, graph, vehicle, sdf, 'a' * 64)
    return inputs


# 功能：
#   核对取件点、返航点和载荷质量均由当前输入编译，四个物理动作保留依赖及单次挂载限制。
# 输入：
#   curriculum_inputs：隔离物理资产与路线。
# 输出：
#   None：绑定或动作约束不符则测试失败。
def test_payload_curriculum_binds_physics_and_order(curriculum_inputs):
    result = build_payload_teacher_curriculum(*curriculum_inputs)
    checkpoints = result['checkpoints'].checkpoints
    assert [item.track_point_index for item in checkpoints] == [1, 2]
    assert checkpoints[0].target_node == curriculum_inputs[0].node_ids[1]
    steps = result['actions'].steps
    assert [step.task_id for step in steps] == [
        'precontact', 'attach-fixture', 'custody', 'loaded-stability']
    assert all(step.checkpoint_id == checkpoints[0].checkpoint_id for step in steps)
    assert steps[1].max_attempts == 1
    assert steps[1].action == 'training.payload-attach-fixture'
    assert steps[2].parameters['payload_mass_kg'] == pytest.approx(.1)
    assert steps[3].depends_on == ['custody']
    assert all(step.driver != 'ros2-service' for step in steps)
    assert 'simulation-only' in result['mission'].constraints
    _validate_execution_contract_binding(result['checkpoints'], result['actions'],
        curriculum_inputs[0], result['mission'].contract_id, True)
    with pytest.raises(ValueError, match='EXPLICIT_TEACHER_COLLECTION'):
        _validate_execution_contract_binding(result['checkpoints'], result['actions'],
            curriculum_inputs[0], result['mission'].contract_id, False)


# 功能：
#   检查仿真适配器拒绝重复、错位、越界或跨任务的检查点，即使每份 JSON 分别合法。
# 输入：
#   curriculum_inputs：当前路线与资产；damage：本轮错误绑定。
# 输出：
#   None：错误合同不能通过起飞前绑定检查。
@pytest.mark.parametrize('damage', ['duplicate', 'index', 'target', 'mission', 'action-target'])
def test_adapter_rejects_cross_file_checkpoint_mismatch(curriculum_inputs, damage):
    result = build_payload_teacher_curriculum(*curriculum_inputs)
    checkpoints, actions = result['checkpoints'], result['actions']
    if damage == 'duplicate':
        checkpoints.checkpoints.append(checkpoints.checkpoints[0].model_copy())
    elif damage == 'index':
        checkpoints.checkpoints[0].track_point_index = 99
    elif damage == 'target':
        checkpoints.checkpoints[0].target_node = 'other'
    elif damage == 'mission':
        actions.contract_id = 'other'
    else:
        actions.steps[0].target_node = 'other'
    with pytest.raises(ValueError):
        _validate_execution_contract_binding(checkpoints, actions, curriculum_inputs[0],
                                             result['mission'].contract_id, True)


# 功能：
#   拒绝未闭合、节点位置错配、重复路线、未声明取件点或超出载荷能力的输入。
# 输入：
#   curriculum_inputs：课程输入；damage：本次破坏的绑定。
# 输出：
#   None：非法课程不能编译，否则测试失败。
@pytest.mark.parametrize('damage', [
    'open', 'position', 'node', 'semantic', 'mass', 'semantic-hash'])
def test_payload_curriculum_rejects_invalid_inputs(curriculum_inputs, damage):
    route, graph, vehicle, sdf, digest = curriculum_inputs
    graph = graph.model_copy(deep=True)
    if damage == 'open':
        route = route.model_copy(update={'goal_node': 'other'})
    elif damage == 'position':
        graph.nodes[1].position_m.x += 1.
    elif damage == 'node':
        graph.nodes.append(graph.nodes[0].model_copy(deep=True))
    elif damage == 'semantic':
        graph.nodes[1].semantic = 'corridor'
    elif damage == 'mass':
        vehicle.max_pickup_payload_kg = .05
    else:
        digest = 'bad'
    with pytest.raises((ValueError, RuntimeError)):
        build_payload_teacher_curriculum(route, graph, vehicle, sdf, digest)


# 功能：
#   在任何飞控连接之前拒绝教师检查点绕过独立安全、加载模型授权或缺少物理动作合同。
# 输入：
#   missing：本次缺失或冲突的启动条件。
# 输出：
#   None：执行器必须立即拒绝，不能触及飞控。
@pytest.mark.parametrize('missing', ['teacher', 'safety', 'model', 'checkpoint', 'action'])
def test_executor_rejects_incomplete_teacher_payload_mode(missing):
    args = SimpleNamespace(teacher_payload_checkpoints=True, simulation_teacher_control=True,
        require_model_control_authority=False, local_safety_required=True,
        checkpoint_contract='checkpoint', runtime_action_contract='action', setpoint_rate_hz=50)
    key, value = {'teacher': ('simulation_teacher_control', False),
        'safety': ('local_safety_required', False),
        'model': ('require_model_control_authority', True),
        'checkpoint': ('checkpoint_contract', None),
        'action': ('runtime_action_contract', None)}[missing]
    setattr(args, key, value)
    with pytest.raises(ValueError, match='exclusive recorded simulation'):
        asyncio.run(executor._run(args, None))
