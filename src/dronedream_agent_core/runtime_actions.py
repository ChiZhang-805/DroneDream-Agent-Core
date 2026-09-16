"""Build hash-bound runtime action contracts from plugin declarations and task DAGs."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import jsonschema
import numpy as np

from dronedream_plugin_sdk.protocol import copy_json, decode_json, validate_local_schema

from .contracts import (
    DomainActionCatalog,
    MissionContract,
    RuntimeActionAdapterCatalog,
    RuntimeActionAdapterDefinition,
    RuntimeActionExecutionContract,
    RuntimeActionExecutionStep,
    RuntimeCheckpointContract,
    TaskGraph,
    VehicleAsset,
)
from .domain_actions import action_by_id
from .hashing import sha256_json
from .xml_values import parse_xml

_MAX_ASSET_BYTES = 8 * 1024 * 1024


class RuntimeActionContractError(RuntimeError):
    """A prepared task cannot be bound to an executable runtime adapter."""


# 功能：
#   合并相同适配器声明，拒绝重复执行器的冲突归属，并阻止契约校验隐式访问外部引用。
# 输入：
#   values：插件返回的运行时动作适配器声明包。
# 输出：
#   catalog：内容身份固定、按适配器标识排序的独立目录。
def merge_runtime_action_adapters(values: list[Any]) -> RuntimeActionAdapterCatalog:
    adapters: dict[str, RuntimeActionAdapterDefinition] = {}
    executor_owners: dict[str, str] = {}
    for raw_pack in values:
        if not isinstance(raw_pack, dict) or not isinstance(raw_pack.get("adapters"), list):
            raise RuntimeActionContractError("RUNTIME_ACTION_ADAPTER_PACK_INVALID")
        try:
            raw_pack = copy_json(raw_pack)
        except ValueError as error:
            raise RuntimeActionContractError("RUNTIME_ACTION_ADAPTER_PACK_INVALID") from error
        for raw_adapter in raw_pack["adapters"]:
            adapter = RuntimeActionAdapterDefinition.model_validate(raw_adapter)
            try:
                adapter.parameter_schema = validate_local_schema(adapter.parameter_schema)
            except (jsonschema.SchemaError, ValueError) as error:
                raise RuntimeActionContractError(
                    f"RUNTIME_ACTION_ADAPTER_SCHEMA_INVALID:{adapter.adapter_id}"
                ) from error
            previous = adapters.get(adapter.adapter_id)
            if previous is not None and previous != adapter:
                raise RuntimeActionContractError(
                    f"RUNTIME_ACTION_ADAPTER_CONFLICT:{adapter.adapter_id}"
                )
            for executor in adapter.runtime_executors:
                owner = executor_owners.get(executor)
                if owner is not None and owner != adapter.adapter_id:
                    raise RuntimeActionContractError(
                        f"RUNTIME_ACTION_EXECUTOR_CONFLICT:{executor}:{owner}:{adapter.adapter_id}"
                    )
                executor_owners[executor] = adapter.adapter_id
            adapters[adapter.adapter_id] = adapter
    if not adapters:
        raise RuntimeActionContractError("RUNTIME_ACTION_ADAPTER_CATALOG_EMPTY")
    ordered = [adapters[adapter_id] for adapter_id in sorted(adapters)]
    payload = [item.model_dump(mode="json") for item in ordered]
    catalog = RuntimeActionAdapterCatalog(
        catalog_id=f"runtime-actions.{sha256_json(payload)[:24]}", adapters=ordered
    )
    return catalog


# 功能：
#   提取 XML 本地标签名，使有无命名空间的 SDF 使用相同语义判断。
# 输入：
#   tag：XML 完整标签名。
# 输出：
#   name：不含命名空间的标签名。
def _local(tag: str) -> str:
    name = tag.rsplit("}", 1)[-1]
    return name


# 功能：
#   有界读取当前资产的实际字节，拒绝超大文件，不执行 SDF 插件。
# 输入：
#   path：准备绑定的资产文件。
# 输出：
#   content：供摘要及解析共同使用的文件字节快照。
def _asset_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(_MAX_ASSET_BYTES + 1)
    if len(content) > _MAX_ASSET_BYTES:
        raise ValueError("RUNTIME_ACTION_ASSET_SIZE_LIMIT")
    return content


# 功能：
#   从唯一可拆卸接头读取实际主题与连接身份，拒绝多插件或重复字段的含糊绑定。
# 输入：
#   vehicle_sdf：当前载具 SDF 路径。
#   vehicle_sdf_bytes：可选的同次编译字节快照，避免重新读取已被替换的文件。
# 输出：
#   bindings：父子连接身份、挂接主题、脱离主题及状态主题。
def _detachable_joint_bindings(
    vehicle_sdf: Path, vehicle_sdf_bytes: bytes | None = None
) -> dict[str, str]:
    try:
        content = _asset_bytes(vehicle_sdf) if vehicle_sdf_bytes is None else vehicle_sdf_bytes
        root = parse_xml(content, maximum_bytes=_MAX_ASSET_BYTES)
    except (ElementTree.ParseError, OSError, ValueError) as error:
        raise RuntimeActionContractError("RUNTIME_ACTION_VEHICLE_SDF_INVALID") from error
    plugins = [
        element
        for element in root.iter()
        if _local(element.tag) == "plugin"
        and "detachable"
        in (f"{element.attrib.get('name', '')} {element.attrib.get('filename', '')}").casefold()
    ]
    if not plugins:
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_PLUGIN_MISSING")
    if len(plugins) != 1:
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_PLUGIN_AMBIGUOUS")
    plugin = plugins[0]
    names = [_local(element.tag) for element in plugin]
    if len(names) != len(set(names)):
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_BINDING_AMBIGUOUS")
    values = {
        _local(element.tag): (element.text or "").strip()
        for element in plugin
        if (element.text or "").strip()
    }
    required = {
        "parent_link",
        "child_model",
        "child_link",
        "attach_topic",
        "detach_topic",
        "output_topic",
    }
    if not required <= values.keys():
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_BINDING_INCOMPLETE")
    bindings = {name: values[name] for name in sorted(required)}
    return bindings


# 功能：
#   将挂载位置、载具/载荷名称与同一份 SDF 和说明文件字节绑定，不把声明视为已挂载。
# 输入：
#   vehicle_sdf：当前载具 SDF 路径。
#   vehicle_sdf_bytes：可选的同次编译 SDF 快照。
# 输出：
#   binding：载荷挂载偏移、误差上限及来源摘要。
def _payload_mount_binding(
    vehicle_sdf: Path, vehicle_sdf_bytes: bytes | None = None
) -> dict[str, Any]:
    summary_path = vehicle_sdf.with_name("summary.json")
    try:
        if vehicle_sdf_bytes is None:
            vehicle_sdf_bytes = _asset_bytes(vehicle_sdf)
        vehicle_root = parse_xml(vehicle_sdf_bytes, maximum_bytes=_MAX_ASSET_BYTES)
        summary_bytes = _asset_bytes(summary_path)
        summary = decode_json(summary_bytes, limit=_MAX_ASSET_BYTES)
    except (ElementTree.ParseError, OSError, ValueError) as error:
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_MOUNT_BINDING_INVALID") from error
    if not isinstance(summary, dict) or not isinstance(summary.get("mission_payload"), dict):
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_MOUNT_BINDING_INVALID")
    model = next(
        (element for element in vehicle_root.iter() if _local(element.tag) == "model"),
        None,
    )
    bindings = _detachable_joint_bindings(vehicle_sdf, vehicle_sdf_bytes)
    payload_summary = summary["mission_payload"]
    try:
        center = payload_summary["center_above_model_root_m"]
        maximum_error = payload_summary.get("maximum_attachment_error_m", 0.02)
        if type(center) not in (int, float) or type(maximum_error) not in (int, float):
            raise ValueError("mount values must be numbers")
        center_above_root = float(center)
        maximum_error_m = float(maximum_error)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_MOUNT_BINDING_INVALID") from error
    vehicle_model_name = "" if model is None else str(model.attrib.get("name", "")).strip()
    payload_model_name = str(payload_summary.get("model_name", "")).strip()
    if (
        not vehicle_model_name
        or payload_model_name != bindings["child_model"]
        or not math.isfinite(center_above_root)
        or not math.isfinite(maximum_error_m)
        or not 0.0 < center_above_root <= 1.0
        or not 0.0 < maximum_error_m <= 0.1
    ):
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_MOUNT_BINDING_INVALID")
    binding_payload = {
        "vehicle_model_name": vehicle_model_name,
        "payload_model_name": payload_model_name,
        "payload_mount_offset_model_m": [0.0, 0.0, center_above_root],
        "payload_mount_max_alignment_error_m": maximum_error_m,
        "vehicle_sdf_sha256": hashlib.sha256(vehicle_sdf_bytes).hexdigest(),
        "vehicle_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
    }
    binding = {
        **binding_payload,
        "payload_mount_binding_sha256": sha256_json(binding_payload),
    }
    return binding


# 功能：
#   从单一载荷惯性声明读取质量和惯量，校验有限性、正定性及主惯量三角约束。
#   这些是供后续读回核对的物理声明，不证明挂载成功或飞控已经适应负载。
# 输入：
#   vehicle_sdf：用于定位同目录载荷 SDF 的载具路径。
# 输出：
#   binding：载荷质量、惯性张量系数及其 SDF 字节摘要。
def _payload_physics_binding(vehicle_sdf: Path) -> dict[str, Any]:
    payload_sdf = vehicle_sdf.with_name("takeout-payload.sdf")
    try:
        payload_bytes = _asset_bytes(payload_sdf)
        root = parse_xml(payload_bytes, maximum_bytes=_MAX_ASSET_BYTES)
    except (ElementTree.ParseError, OSError, ValueError) as error:
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_SDF_INVALID") from error

    inertials = [element for element in root.iter() if _local(element.tag) == "inertial"]
    if not inertials:
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_INERTIAL_MISSING")
    if len(inertials) != 1:
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_INERTIAL_AMBIGUOUS")
    inertial = inertials[0]
    names = [_local(element.tag) for element in inertial.iter() if (element.text or "").strip()]
    if len(names) != len(set(names)):
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_INERTIAL_AMBIGUOUS")
    values = {
        _local(element.tag): (element.text or "").strip()
        for element in inertial.iter()
        if (element.text or "").strip()
    }
    required = {"mass", "ixx", "iyy", "izz", "ixy", "ixz", "iyz"}
    if not required <= values.keys():
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_INERTIAL_INCOMPLETE")
    try:
        mass_kg = float(values["mass"])
        inertia = {name: float(values[name]) for name in sorted(required - {"mass"})}
    except ValueError as error:
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_INERTIAL_INVALID") from error
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_MASS_INVALID")
    if not all(math.isfinite(value) for value in inertia.values()) or any(
        inertia[name] <= 0.0 for name in ("ixx", "iyy", "izz")
    ):
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_INERTIA_INVALID")
    # 正的对角线不足以证明惯性张量可实现；先缩放再求三个主惯量，避免大值溢出。
    scale = max(abs(value) for value in inertia.values())
    tensor = (
        np.array(
            [
                [inertia["ixx"], inertia["ixy"], inertia["ixz"]],
                [inertia["ixy"], inertia["iyy"], inertia["iyz"]],
                [inertia["ixz"], inertia["iyz"], inertia["izz"]],
            ],
            dtype=np.float64,
        )
        / scale
    )
    principal = np.linalg.eigvalsh(tensor)
    if principal[0] <= 0.0 or principal[2] > principal[0] + principal[1] + 1e-12:
        raise RuntimeActionContractError("RUNTIME_ACTION_PAYLOAD_INERTIA_INVALID")
    binding = {
        "payload_sdf_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload_mass_kg": mass_kg,
        "payload_inertia_kg_m2": inertia,
    }
    return binding


# 功能：
#   按依赖优先排列已经验证的任务图，同层按任务标识确定稳定顺序。
# 输入：
#   graph：已通过唯一性、依赖存在性和无环校验的任务图。
# 输出：
#   ordered：所有前置任务均先于后继的节点列表。
def _topological_nodes(graph: TaskGraph):
    by_id = {node.task_id: node for node in graph.nodes}
    remaining = {node.task_id: set(node.depends_on) for node in graph.nodes}
    ordered = []
    while remaining:
        roots = sorted(task_id for task_id, dependencies in remaining.items() if not dependencies)
        if not roots:
            raise RuntimeActionContractError("RUNTIME_ACTION_TASK_GRAPH_CYCLIC")
        for task_id in roots:
            ordered.append(by_id[task_id])
            del remaining[task_id]
        for dependencies in remaining.values():
            dependencies.difference_update(roots)
    return ordered


# 功能：
#   遍历任务的传递前置关系，用于证明动作真正依赖对应地点的到达事件。
# 输入：
#   graph：已经验证的任务图。
#   task_id：待查询的任务标识。
# 输出：
#   result：去重后的所有前置任务标识。
def _ancestors(graph: TaskGraph, task_id: str) -> set[str]:
    by_id = {node.task_id: node for node in graph.nodes}
    result: set[str] = set()
    pending = list(by_id[task_id].depends_on)
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        result.add(dependency)
        pending.extend(by_id[dependency].depends_on)
    return result


# 功能：
#   按执行器标识建立索引，调用前必须已拒绝声明中的所有权冲突。
# 输入：
#   catalog：已验证的运行时动作适配器目录。
# 输出：
#   adapters：执行器标识到唯一适配器的映射。
def _adapter_by_executor(
    catalog: RuntimeActionAdapterCatalog,
) -> dict[str, RuntimeActionAdapterDefinition]:
    adapters = {
        executor: adapter for adapter in catalog.adapters for executor in adapter.runtime_executors
    }
    return adapters


# 功能：
#   把领域动作解析成设备参数，阻止任务参数覆盖实际主题、服务身份或载荷物理约束。
#   复制默认值和任务参数，避免返回值改写原任务及插件目录。
# 输入：
#   driver：动作所绑定的运行时驱动类型。
#   runtime_executor：精确的运行时执行器标识。
#   defaults：适配器的默认参数。
#   arguments：任务节点的业务参数。
#   vehicle_sdf：当前载具的实际 SDF 路径。
# 输出：
#   parameters：与所选驱动匹配的独立参数对象。
def _resolved_parameters(
    *,
    driver: str,
    runtime_executor: str,
    defaults: dict[str, Any],
    arguments: dict[str, Any],
    vehicle_sdf: Path,
) -> dict[str, Any]:
    # A caller editing a resolved parameter must not mutate the task graph or
    # shared plugin defaults after their hashes have been frozen.
    defaults, arguments = deepcopy(defaults), deepcopy(arguments)
    if driver == "gazebo-payload":
        operations = {"native.payload.pickup": "attach", "native.payload.release": "detach"}
        operation = operations.get(runtime_executor)
        if operation is None:
            raise RuntimeActionContractError(
                f"RUNTIME_ACTION_PAYLOAD_EXECUTOR_UNSUPPORTED:{runtime_executor}"
            )
        protected = {
            "protocol",
            "operation",
            "topic",
            "output_topic",
            "vehicle_model_name",
            "payload_model_name",
            "payload_mount_offset_model_m",
            "payload_mount_max_alignment_error_m",
            "payload_mount_binding_sha256",
        }
        if protected & arguments.keys():
            raise RuntimeActionContractError(
                f"RUNTIME_ACTION_PROTECTED_PARAMETER_OVERRIDE:{runtime_executor}"
            )
        try:
            vehicle_bytes = _asset_bytes(vehicle_sdf)
        except (OSError, ValueError) as error:
            raise RuntimeActionContractError("RUNTIME_ACTION_VEHICLE_SDF_INVALID") from error
        bindings = _detachable_joint_bindings(vehicle_sdf, vehicle_bytes)
        topic_key = "attach_topic" if operation == "attach" else "detach_topic"
        parameters = {
            **defaults,
            **_payload_mount_binding(vehicle_sdf, vehicle_bytes),
            "protocol": "gazebo-transport",
            "operation": operation,
            "topic": bindings[topic_key],
            "output_topic": bindings["output_topic"],
            "arguments": arguments,
        }
        return parameters
    if driver == "payload-transition":
        protected = {
            "operation",
            "output_topic",
            "payload_sdf_sha256",
            "payload_mass_kg",
            "payload_inertia_kg_m2",
            "position_tolerance_m",
            "speed_tolerance_mps",
            "stable_window_seconds",
            "settle_timeout_seconds",
        }
        if protected & arguments.keys():
            raise RuntimeActionContractError(
                f"RUNTIME_ACTION_PROTECTED_PARAMETER_OVERRIDE:{runtime_executor}"
            )
        operations = {
            "native.payload.precontact-hold": "precontact",
            "native.payload.confirm-custody": "confirm-custody",
            "native.payload.verify-loaded-stability": "postattach-stability",
        }
        operation = operations.get(runtime_executor)
        if operation is None:
            raise RuntimeActionContractError(
                f"RUNTIME_ACTION_PAYLOAD_TRANSITION_UNSUPPORTED:{runtime_executor}"
            )
        bindings = _detachable_joint_bindings(vehicle_sdf)
        parameters = {
            **defaults,
            **_payload_physics_binding(vehicle_sdf),
            "operation": operation,
            "output_topic": bindings["output_topic"],
            "arguments": arguments,
        }
        return parameters
    if driver == "mavsdk-camera":
        protected = {"command"}
        if protected & arguments.keys():
            raise RuntimeActionContractError(
                f"RUNTIME_ACTION_PROTECTED_PARAMETER_OVERRIDE:{runtime_executor}"
            )
        component_id = arguments.get("component_id", defaults.get("component_id", 100))
        if type(component_id) is not int or not 1 <= component_id <= 255:
            raise RuntimeActionContractError("RUNTIME_ACTION_CAMERA_COMPONENT_ID_INVALID")
        remaining = {key: value for key, value in arguments.items() if key != "component_id"}
        parameters = {
            **defaults,
            "command": "take_photo",
            "component_id": component_id,
            "arguments": remaining,
        }
        return parameters
    if driver == "ros2-service":
        protected = {"service_name", "service_type"}
        if protected & arguments.keys():
            raise RuntimeActionContractError(
                f"RUNTIME_ACTION_PROTECTED_PARAMETER_OVERRIDE:{runtime_executor}"
            )
        namespace = str(defaults.get("service_namespace", "/dronedream/domain_actions")).rstrip("/")
        suffix = runtime_executor.removeprefix("native.").replace(".", "/").replace("-", "_")
        try:
            arguments_json = json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (ValueError, TypeError, RecursionError) as error:
            raise RuntimeActionContractError("RUNTIME_ACTION_ARGUMENTS_INVALID") from error
        parameters = {
            "service_name": f"{namespace}/{suffix}",
            "service_type": str(
                defaults.get("service_type", "dronedream_agent_msgs/srv/ExecuteDomainAction")
            ),
            "request": {
                "contract_id": "",
                "task_id": "",
                "action": runtime_executor,
                "target_node": "",
                "arguments_json": arguments_json,
            },
        }
        return parameters
    raise RuntimeActionContractError(f"RUNTIME_ACTION_DRIVER_UNSUPPORTED:{driver}")


# 功能：
#   将非移动业务任务编译成绑定真实前置到达事件和设备声明的运行步骤。
#   入口重验可变合同、图与检查点；移动仍由独立控制链执行，编译不发送设备命令。
# 输入：
#   mission_contract：用户确认后冻结的任务合同。
#   task_graph：包含动作、依赖及目标地点的任务图。
#   domain_actions：任务可使用的动作目录。
#   adapter_catalog：动作到实际设备驱动的适配器目录。
#   checkpoints：本合同对应的路径到达检查点。
#   vehicle：当前载具几何、质量及载荷限制。
#   vehicle_sdf：当前载具的实际 SDF 路径。
# 输出：
#   execution_contract：绑定合同及各输入摘要的有序运行时动作契约。
def build_runtime_action_execution_contract(
    *,
    mission_contract: MissionContract,
    task_graph: TaskGraph,
    domain_actions: DomainActionCatalog,
    adapter_catalog: RuntimeActionAdapterCatalog,
    checkpoints: RuntimeCheckpointContract,
    vehicle: VehicleAsset,
    vehicle_sdf: Path,
) -> RuntimeActionExecutionContract:
    # Pydantic 的 model_copy 和嵌套列表原地编辑不会重新验证，不能只依赖构造时检查。
    mission_contract = MissionContract.model_validate(
        mission_contract.model_dump(mode="python"), strict=True
    )
    task_graph = TaskGraph.model_validate(task_graph.model_dump(mode="python"), strict=True)
    domain_actions = DomainActionCatalog.model_validate(
        domain_actions.model_dump(mode="python"), strict=True
    )
    adapter_catalog = RuntimeActionAdapterCatalog.model_validate(
        adapter_catalog.model_dump(mode="python"), strict=True
    )
    checkpoints = RuntimeCheckpointContract.model_validate(
        checkpoints.model_dump(mode="python"), strict=True
    )
    vehicle = VehicleAsset.model_validate(vehicle.model_dump(mode="python"), strict=True)
    if checkpoints.contract_id != mission_contract.contract_id:
        raise RuntimeActionContractError("RUNTIME_ACTION_CHECKPOINT_CONTRACT_MISMATCH")
    task_by_id = {task.task_id: task for task in task_graph.nodes}
    seen_checkpoints, seen_tasks = set(), set()
    for checkpoint in checkpoints.checkpoints:
        task = task_by_id.get(checkpoint.task_id)
        if (
            checkpoint.checkpoint_id in seen_checkpoints
            or checkpoint.task_id in seen_tasks
            or task is None
            or task.target_node != checkpoint.target_node
            or not action_by_id(domain_actions, task.action).movement
        ):
            raise RuntimeActionContractError("RUNTIME_ACTION_CHECKPOINT_BINDING_INVALID")
        seen_checkpoints.add(checkpoint.checkpoint_id)
        seen_tasks.add(checkpoint.task_id)
    # 目录可能在构造后被插件修改，编译时再次禁止远程 Schema 引用。
    for adapter in adapter_catalog.adapters:
        try:
            adapter.parameter_schema = validate_local_schema(adapter.parameter_schema)
        except (jsonschema.SchemaError, ValueError) as error:
            raise RuntimeActionContractError("RUNTIME_ACTION_ADAPTER_SCHEMA_INVALID") from error
    adapters = _adapter_by_executor(adapter_catalog)
    runtime_action_task_ids = {
        task.task_id
        for task in task_graph.nodes
        if not action_by_id(domain_actions, task.action).movement
        and action_by_id(domain_actions, task.action).flight_boundary == "none"
        and task.action != "none"
    }
    checkpoint_order = {
        checkpoint.checkpoint_id: index for index, checkpoint in enumerate(checkpoints.checkpoints)
    }
    checkpoint_by_task = {checkpoint.task_id: checkpoint for checkpoint in checkpoints.checkpoints}

    steps: list[RuntimeActionExecutionStep] = []
    for task in _topological_nodes(task_graph):
        definition = action_by_id(domain_actions, task.action)
        if definition.movement or definition.flight_boundary != "none" or task.action == "none":
            continue
        runtime_executor = definition.runtime_executor
        if runtime_executor is None:
            raise RuntimeActionContractError(f"RUNTIME_ACTION_EXECUTOR_MISSING:{task.action}")
        adapter = adapters.get(runtime_executor)
        if adapter is None:
            raise RuntimeActionContractError(
                f"RUNTIME_ACTION_ADAPTER_MISSING:{task.action}:{runtime_executor}"
            )
        ancestors = _ancestors(task_graph, task.task_id)
        ancestor_checkpoints = [
            checkpoint_by_task[task_id]
            for task_id in ancestors
            if task_id in checkpoint_by_task
            and checkpoint_by_task[task_id].target_node == task.target_node
        ]
        # 同一地点的未来或无关检查点不证明本动作已经到达，禁止按地点名回退借用。
        candidates = ancestor_checkpoints
        checkpoint = (
            max(candidates, key=lambda item: checkpoint_order[item.checkpoint_id])
            if candidates
            else None
        )
        if checkpoint is None and task.target_node != mission_contract.start_node:
            raise RuntimeActionContractError(
                f"RUNTIME_ACTION_TRIGGER_UNRESOLVED:{task.task_id}:{task.target_node}"
            )
        parameters = _resolved_parameters(
            driver=adapter.driver,
            runtime_executor=runtime_executor,
            defaults=adapter.default_parameters,
            arguments=task.arguments,
            vehicle_sdf=vehicle_sdf,
        )
        if adapter.driver == "payload-transition":
            # Respect both the payload mount rating and remaining takeoff mass;
            # a small attachment is not automatically safe for every vehicle.
            payload_mass_kg = float(parameters["payload_mass_kg"])
            permitted_payload_kg = min(
                vehicle.max_pickup_payload_kg,
                max(0.0, vehicle.max_takeoff_mass_kg - vehicle.dry_mass_kg),
            )
            if payload_mass_kg > permitted_payload_kg + 1e-9:
                raise RuntimeActionContractError(
                    f"RUNTIME_ACTION_PAYLOAD_EXCEEDS_VEHICLE_CAPACITY:{task.task_id}"
                )
        if adapter.driver == "ros2-service":
            parameters["request"].update(
                {
                    "contract_id": mission_contract.contract_id,
                    "task_id": task.task_id,
                    "target_node": task.target_node,
                }
            )
        try:
            if adapter.parameter_schema:
                jsonschema.validate(parameters, adapter.parameter_schema)
        except jsonschema.ValidationError as error:
            raise RuntimeActionContractError(
                f"RUNTIME_ACTION_PARAMETERS_INVALID:{task.task_id}:{error.validator}"
            ) from error
        steps.append(
            RuntimeActionExecutionStep(
                step_id=f"action-{len(steps) + 1:03d}",
                task_id=task.task_id,
                action=task.action,
                target_node=task.target_node,
                trigger="checkpoint" if checkpoint is not None else "post-takeoff",
                checkpoint_id=checkpoint.checkpoint_id if checkpoint is not None else None,
                # Flight-boundary and movement prerequisites are enforced by
                # the trigger itself. Keep only action-to-action dependencies
                # in the portable runtime contract.
                depends_on=[
                    dependency
                    for dependency in task.depends_on
                    if dependency in runtime_action_task_ids
                ],
                runtime_executor=runtime_executor,
                adapter_id=adapter.adapter_id,
                driver=adapter.driver,
                parameters=parameters,
                required_success_evidence=definition.required_success_evidence,
                # A Gazebo attach/detach driver already publishes the same
                # command repeatedly inside one bounded observation window.
                # If its state remains unknown, starting the whole driver again
                # could move a child that was physically attached after its
                # one-shot event was lost. Non-idempotent physical transitions
                # therefore get one driver attempt; higher-level recovery must
                # inspect fresh state before issuing a new mission contract.
                max_attempts=(1 if adapter.driver == "gazebo-payload" else task.max_retries + 1),
                fallback=task.fallback,
                timeout_seconds=adapter.timeout_seconds,
                authority=adapter.authority,
            )
        )
    execution_contract = RuntimeActionExecutionContract(
        contract_id=mission_contract.contract_id,
        task_graph_sha256=sha256_json(task_graph),
        domain_action_catalog_sha256=sha256_json(domain_actions),
        adapter_catalog_sha256=sha256_json(adapter_catalog),
        steps=steps,
    )
    return execution_contract
