"""动作编译必须匹配真实任务依赖、设备声明和物理载荷，不从名称推断执行资格。"""

import json

import pytest
from test_domain_actions import _catalog, _contract, _vehicle
from test_runtime_commands import _ack, _decision, _message

from dronedream_agent_core import runtime_actions
from dronedream_agent_core.contracts import RuntimeCheckpoint, RuntimeCheckpointContract, TaskGraph
from dronedream_agent_core.domain_actions import DomainActionError, merge_action_packs
from dronedream_agent_core.plugin_api import build_discovered_extension_registry
from dronedream_agent_core.runtime_commands import RuntimeCommandError, build_runtime_command


# 功能：
#   建立非载荷动作的完整编译输入，用实际插件声明而非固定编译返回值进行测试。
# 输入：
#   tmp_path：当前测试隔离目录。
# 输出：
#   inputs：任务合同、图、检查点、载具及动作适配器。
@pytest.fixture
def compile_inputs(tmp_path):
    catalog = _catalog()
    contract = _contract(catalog)
    registry = build_discovered_extension_registry()
    outputs, _ = registry.invoke_multiple(
        "runtime.action-adapters", "declare_runtime_action_adapters"
    )
    graph = TaskGraph.model_validate(
        {
            "nodes": [
                {
                    "task_id": "launch",
                    "action": "takeoff",
                    "target_node": "office",
                    "success_evidence": ["airborne"],
                    "fallback": "land",
                },
                {
                    "task_id": "navigate",
                    "action": "navigate",
                    "target_node": "pickup",
                    "depends_on": ["launch"],
                    "success_evidence": ["arrived"],
                    "fallback": "hold",
                },
                {
                    "task_id": "inspect",
                    "action": "inspection.capture-thermal",
                    "target_node": "pickup",
                    "depends_on": ["navigate"],
                    "success_evidence": ["thermal"],
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
            )
        ],
    )
    inputs = dict(
        mission_contract=contract,
        task_graph=graph,
        domain_actions=catalog,
        adapter_catalog=runtime_actions.merge_runtime_action_adapters(outputs),
        checkpoints=checkpoints,
        vehicle=_vehicle(),
        vehicle_sdf=tmp_path / "unused.sdf",
    )
    return inputs


# 功能：
#   验证具有实际前置到达关系的动作仍然可以编译，不把新增边界变成全面禁用。
# 输入：
#   compile_inputs：完整的离线任务编译输入。
# 输出：
#   None：不返回业务数据。
def test_valid_action_keeps_its_actual_arrival_dependency(compile_inputs):
    result = runtime_actions.build_runtime_action_execution_contract(**compile_inputs)
    assert len(result.steps) == 1
    assert result.steps[0].checkpoint_id == "checkpoint-001"


# 功能：
#   验证同地点但无前置依赖的检查点不能给动作提供到达资格。
# 输入：
#   compile_inputs：完整的离线任务编译输入。
# 输出：
#   None：不返回业务数据。
def test_unrelated_same_location_checkpoint_cannot_authorize_action(compile_inputs):
    compile_inputs["task_graph"].nodes[-1].depends_on.clear()
    with pytest.raises(runtime_actions.RuntimeActionContractError, match="TRIGGER_UNRESOLVED"):
        runtime_actions.build_runtime_action_execution_contract(**compile_inputs)


# 功能：
#   验证检查点的合同、任务身份、目标和唯一性都属于编译入口约束。
# 输入：
#   compile_inputs：完整的离线任务编译输入。
#   mutation：注入的检查点错配。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["contract", "target", "task", "duplicate"])
def test_checkpoint_binding_is_revalidated_before_compilation(compile_inputs, mutation):
    checkpoints = compile_inputs["checkpoints"]
    if mutation == "contract":
        checkpoints.contract_id = "another-contract"
    elif mutation == "target":
        checkpoints.checkpoints[0].target_node = "office"
    elif mutation == "task":
        checkpoints.checkpoints[0].task_id = "unknown-task"
    else:
        checkpoints.checkpoints.append(checkpoints.checkpoints[0].model_copy(deep=True))
    with pytest.raises(runtime_actions.RuntimeActionContractError, match="CHECKPOINT"):
        runtime_actions.build_runtime_action_execution_contract(**compile_inputs)


# 功能：
#   验证动作与适配器契约拒绝远程 Schema 引用，不能在校验过程中隐式访问网络。
# 输入：
#   kind：动作或适配器声明分支。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["action", "adapter"])
def test_action_schema_must_be_self_contained(kind):
    if kind == "action":
        catalog = _catalog()
        action = next(item for item in catalog.actions if item.action_id == "takeoff")
        raw = action.model_dump(mode="json")
        raw["input_schema"] = {"$ref": "https://example.invalid/external.json"}
        with pytest.raises(DomainActionError, match="SCHEMA_INVALID"):
            merge_action_packs([{"domain_id": action.domain_id, "actions": [raw]}])
    else:
        with pytest.raises(runtime_actions.RuntimeActionContractError, match="SCHEMA_INVALID"):
            runtime_actions.merge_runtime_action_adapters(
                [
                    {
                        "adapters": [
                            {
                                "adapter_id": "camera",
                                "runtime_executors": ["native.camera.capture"],
                                "driver": "mavsdk-camera",
                                "authority": "control",
                                "parameter_schema": {
                                    "$ref": "https://example.invalid/external.json"
                                },
                            }
                        ]
                    }
                ]
            )


# 功能：
#   构造已声明名称和挂载位置的载具文件，供载荷绑定边界测试使用。
# 输入：
#   tmp_path：当前测试隔离目录。
# 输出：
#   path：测试载具 SDF 文件路径。
@pytest.fixture
def vehicle_sdf(tmp_path):
    path = tmp_path / "vehicle.sdf"
    path.write_text(
        """<sdf version="1.10"><model name="drone"><plugin name="detachable_joint">
        <parent_link>base_link</parent_link><child_model>payload</child_model>
        <child_link>payload_link</child_link><attach_topic>/payload/attach</attach_topic>
        <detach_topic>/payload/detach</detach_topic><output_topic>/payload/state</output_topic>
        </plugin></model></sdf>""",
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "mission_payload": {
                    "model_name": "payload",
                    "center_above_model_root_m": 0.12,
                    "maximum_attachment_error_m": 0.02,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


# 功能：
#   验证多个可拆卸接头声明不能被静默择一绑定，防止选错设备。
# 输入：
#   vehicle_sdf：隔离的载具声明。
# 输出：
#   None：不返回业务数据。
def test_multiple_payload_plugins_are_rejected_as_ambiguous(vehicle_sdf):
    content = vehicle_sdf.read_text(encoding="utf-8")
    plugin = content.split('<plugin name="detachable_joint">')[1].split("</plugin>")[0]
    content = content.replace(
        "</model>", f'<plugin name="detachable_joint">{plugin}</plugin></model>'
    )
    vehicle_sdf.write_text(content, encoding="utf-8")
    with pytest.raises(runtime_actions.RuntimeActionContractError, match="AMBIGUOUS"):
        runtime_actions._detachable_joint_bindings(vehicle_sdf)


# 功能：
#   验证挂载量不能用 JSON 布尔值冒充物理米数。
# 输入：
#   vehicle_sdf：隔离的载具声明。
# 输出：
#   None：不返回业务数据。
def test_payload_mount_height_rejects_boolean(vehicle_sdf):
    summary_path = vehicle_sdf.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["mission_payload"]["center_above_model_root_m"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(runtime_actions.RuntimeActionContractError, match="BINDING_INVALID"):
        runtime_actions._payload_mount_binding(vehicle_sdf)


# 功能：
#   验证非正定惯性和不满足主惯量三角约束的载荷不能通过物理声明校验。
# 输入：
#   vehicle_sdf：隔离的载具路径。
#   diagonal：惯性矩阵三个对角元素。
#   off_diagonal：XY 交叉惯量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("diagonal,off_diagonal", [((1, 1, 1), 2), ((1, 1, 3), 0)])
def test_impossible_payload_inertia_is_rejected(vehicle_sdf, diagonal, off_diagonal):
    ixx, iyy, izz = diagonal
    vehicle_sdf.with_name("takeout-payload.sdf").write_text(
        f"""<sdf><model name="payload">
        <link name="payload_link"><inertial><mass>0.1</mass><inertia>
        <ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz><ixy>{off_diagonal}</ixy>
        <ixz>0</ixz><iyz>0</iyz></inertia></inertial></link></model></sdf>""",
        encoding="utf-8",
    )
    with pytest.raises(runtime_actions.RuntimeActionContractError, match="INERTIA_INVALID"):
        runtime_actions._payload_physics_binding(vehicle_sdf)


# 功能：
#   验证需要修改计划的插件指令不能偷换成直接设备命令。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_plan_revision_requirement_blocks_direct_device_command():
    message = _message()
    ack = _ack(message)
    decision = _decision(message, ack, action="set_avoidance", parameters={"enabled": True})
    decision.amendment_directive.requires_plan_revision = True
    with pytest.raises(RuntimeCommandError, match="GATE_FAILED"):
        build_runtime_command(message=message, acknowledgement=ack, decision=decision)
