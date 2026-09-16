"""Declare action bindings for mission preparation, not execution or device qualification.

Core binds these schemas to the selected asset and records actuator readback.
A declaration alone does not mean a camera, payload mount or ROS service exists.
"""

from __future__ import annotations

from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_plugin_sdk.protocol import copy_json

from ._helpers import hook_plugin


# 功能：
#   冻结适配器声明，每个任务读取时再复制，防止修改某次默认值影响其他任务。
# 输入：
#   adapters：需要冻结的运行动作适配器定义。
# 输出：
#   declare：返回独立声明对象的回调。
def _declare(adapters: list[dict[str, Any]]):
    frozen = copy_json(adapters)

    # 功能：
    #   只返回适配器契约，导入或查询此插件不能触发设备动作。
    # 输入：
    #   _：固定声明不使用的扩展参数。
    # 输出：
    #   declaration：包含独立适配器列表的对象。
    def declare(**_: Any) -> dict[str, Any]:
        declaration = {"adapters": copy_json(frozen)}
        return declaration

    return declare


# 功能：
#   注册兼容执行器及参数契约，声明失败关闭和下一任务切换，不代替实际设备绑定校验。
# 输入：
#   plugin_id：插件标识。
#   name：显示名称。
#   description：适配用途说明。
#   adapters：该插件提供的适配器列表。
#   order：装配及显示顺序。
# 输出：
#   definition：包含声明回调、适配器标识及兼容执行器元数据的插件定义。
def _definition(
    *,
    plugin_id: str,
    name: str,
    description: str,
    adapters: list[dict[str, Any]],
    order: int,
) -> PluginDefinition:
    definition = hook_plugin(
        module_name=__name__,
        plugin_id=plugin_id,
        name=name,
        description=description,
        capability_id=f"{plugin_id}.adapters",
        capability_kind="runtime-adapter",
        capability_name=name,
        capability_description=description,
        category_id="runtime-action",
        category_label="任务动作执行",
        slot_id="runtime.action-adapters",
        slot_label="任务动作适配器",
        activation_mode="multiple",
        category_order=62,
        slot_order=10,
        plugin_order=order,
        hooks={"declare_runtime_action_adapters": _declare(adapters)},
        default_enabled=True,
        failure_mode="fail-closed",
        swap_policy="next-mission",
        metadata={
            "adapter_ids": [str(item["adapter_id"]) for item in adapters],
            "runtime_executors": sorted(
                {str(executor) for item in adapters for executor in item["runtime_executors"]}
            ),
        },
    )
    return definition


# 功能：
#   复制适配器默认参数和 Schema，并声明驱动、所需权限及外层等待预算。
# 输入：
#   adapter_id：适配器标识。
#   executors：允许绑定的运行执行器标识。
#   driver：运行层应选择的驱动类别。
#   defaults：构造实际绑定时使用的默认参数。
#   schema：最终绑定参数的结构约束。
#   authority：动作执行前需要独立核准的权限。
#   timeout_seconds：外层动作等待预算，单位秒。
# 输出：
#   adapter：不共享可变输入容器的适配器声明。
def _adapter(
    *,
    adapter_id: str,
    executors: list[str],
    driver: str,
    defaults: dict[str, Any],
    schema: dict[str, Any],
    authority: str,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    adapter = copy_json(
        {
            "schema_version": "dronedream.runtime-action-adapter.v1",
            "adapter_id": adapter_id,
            "runtime_executors": executors,
            "driver": driver,
            "default_parameters": defaults,
            "parameter_schema": schema,
            "timeout_seconds": timeout_seconds,
            "authority": authority,
        }
    )
    return adapter


# 功能：
#   1. 声明相机、载荷关节、交接稳定门控及 ROS 2 领域服务的适配契约。
#   2. 分开设置内部稳定／读回时限和外层等待余量，不以接口声明证明设备实际存在。
# 输入：
#   无。
# 输出：
#   definitions：内置运行动作适配插件列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions = [
        _definition(
            plugin_id="runtime-actions.camera-mavsdk",
            name="MAVSDK 相机动作",
            description="为可见光拍照声明 MAVSDK 相机接口，执行时仍需设备可用及 MAVLink 确认。",
            order=10,
            adapters=[
                _adapter(
                    adapter_id="runtime.camera.mavsdk",
                    executors=[
                        "native.sensor.capture-rgb",
                    ],
                    driver="mavsdk-camera",
                    defaults={"command": "take_photo", "component_id": 100},
                    schema={
                        "type": "object",
                        "required": ["command", "component_id", "arguments"],
                        "properties": {
                            "command": {"const": "take_photo"},
                            "component_id": {"type": "integer", "minimum": 1, "maximum": 255},
                            "arguments": {"type": "object"},
                        },
                        "additionalProperties": False,
                    },
                    authority="actuate",
                    timeout_seconds=40.0,
                )
            ],
        ),
        _definition(
            plugin_id="runtime-actions.payload-gazebo",
            name="Gazebo 可拆卸载荷动作",
            description="从合格无人机 SDF 读取连接主题，执行取件、释放和投放并验证状态读回。",
            order=20,
            adapters=[
                _adapter(
                    adapter_id="runtime.payload.gazebo-detachable-joint",
                    executors=[
                        "native.payload.pickup",
                        "native.payload.release",
                    ],
                    driver="gazebo-payload",
                    defaults={},
                    schema={
                        "type": "object",
                        "required": [
                            "protocol",
                            "operation",
                            "topic",
                            "output_topic",
                            "vehicle_model_name",
                            "payload_model_name",
                            "payload_mount_offset_model_m",
                            "payload_mount_max_alignment_error_m",
                            "vehicle_sdf_sha256",
                            "vehicle_summary_sha256",
                            "payload_mount_binding_sha256",
                            "arguments",
                        ],
                        "properties": {
                            "protocol": {"const": "gazebo-transport"},
                            "operation": {"enum": ["attach", "detach"]},
                            "topic": {"type": "string", "pattern": "^/"},
                            "output_topic": {"type": "string", "pattern": "^/"},
                            "vehicle_model_name": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "payload_model_name": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "payload_mount_offset_model_m": {
                                "type": "array",
                                "prefixItems": [
                                    {"type": "number"},
                                    {"type": "number"},
                                    {"type": "number"},
                                ],
                                "items": False,
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "payload_mount_max_alignment_error_m": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "maximum": 0.1,
                            },
                            "vehicle_sdf_sha256": {
                                "type": "string",
                                "pattern": "^[0-9a-f]{64}$",
                            },
                            "vehicle_summary_sha256": {
                                "type": "string",
                                "pattern": "^[0-9a-f]{64}$",
                            },
                            "payload_mount_binding_sha256": {
                                "type": "string",
                                "pattern": "^[0-9a-f]{64}$",
                            },
                            "arguments": {"type": "object"},
                        },
                        "additionalProperties": False,
                    },
                    authority="actuate",
                    # The native driver allows up to 15 seconds for Gazebo's
                    # detachable-joint state to converge.  Keep a separate
                    # outer margin for command publication, hold refreshes and
                    # the final state read instead of racing the same deadline.
                    timeout_seconds=25.0,
                )
            ],
        ),
        _definition(
            plugin_id="runtime-actions.payload-transition-guards",
            name="载荷转换状态守卫",
            description=(
                "用 PX4 位置速度遥测和 Gazebo 可拆卸关节读回，分别证明接触前稳定与挂载后物理交接。"
            ),
            order=25,
            adapters=[
                _adapter(
                    adapter_id="runtime.payload.transition-guards",
                    executors=[
                        "native.payload.precontact-hold",
                        "native.payload.confirm-custody",
                    ],
                    driver="payload-transition",
                    defaults={
                        "position_tolerance_m": 0.2,
                        "speed_tolerance_mps": 0.15,
                        # A pickup is a real handoff window, not a momentary
                        # stability sample.  Hold for ten continuous seconds
                        # so a worker can attach the declared payload before
                        # custody and loaded-flight gates run.
                        "stable_window_seconds": 10.0,
                        "settle_timeout_seconds": 25.0,
                    },
                    schema={
                        "type": "object",
                        "required": [
                            "operation",
                            "output_topic",
                            "payload_sdf_sha256",
                            "payload_mass_kg",
                            "payload_inertia_kg_m2",
                            "position_tolerance_m",
                            "speed_tolerance_mps",
                            "stable_window_seconds",
                            "settle_timeout_seconds",
                            "arguments",
                        ],
                        "properties": {
                            "operation": {"enum": ["precontact", "confirm-custody"]},
                            "output_topic": {"type": "string", "pattern": "^/"},
                            "payload_sdf_sha256": {
                                "type": "string",
                                "pattern": "^[0-9a-f]{64}$",
                            },
                            "payload_mass_kg": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "payload_inertia_kg_m2": {
                                "type": "object",
                                "required": ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"],
                                "properties": {
                                    "ixx": {"type": "number", "exclusiveMinimum": 0},
                                    "iyy": {"type": "number", "exclusiveMinimum": 0},
                                    "izz": {"type": "number", "exclusiveMinimum": 0},
                                    "ixy": {"type": "number"},
                                    "ixz": {"type": "number"},
                                    "iyz": {"type": "number"},
                                },
                                "additionalProperties": False,
                            },
                            "position_tolerance_m": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "speed_tolerance_mps": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "stable_window_seconds": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "settle_timeout_seconds": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "arguments": {"type": "object"},
                        },
                        "additionalProperties": False,
                    },
                    authority="control",
                    timeout_seconds=35.0,
                )
            ],
        ),
        _definition(
            plugin_id="runtime-actions.loaded-flight-stability",
            name="挂载后飞行稳定性闸门",
            description=(
                "在载荷关节已确认后保持当前航点，用真实 PX4 位置速度遥测证明新增质量下"
                "仍可稳定悬停。"
            ),
            order=27,
            adapters=[
                _adapter(
                    adapter_id="runtime.payload.loaded-stability",
                    executors=["native.payload.verify-loaded-stability"],
                    driver="payload-transition",
                    defaults={
                        "position_tolerance_m": 0.2,
                        "speed_tolerance_mps": 0.12,
                        "stable_window_seconds": 2.0,
                        "settle_timeout_seconds": 15.0,
                    },
                    schema={
                        "type": "object",
                        "required": [
                            "operation",
                            "output_topic",
                            "payload_sdf_sha256",
                            "payload_mass_kg",
                            "payload_inertia_kg_m2",
                            "position_tolerance_m",
                            "speed_tolerance_mps",
                            "stable_window_seconds",
                            "settle_timeout_seconds",
                            "arguments",
                        ],
                        "properties": {
                            "operation": {"const": "postattach-stability"},
                            "output_topic": {"type": "string", "pattern": "^/"},
                            "payload_sdf_sha256": {
                                "type": "string",
                                "pattern": "^[0-9a-f]{64}$",
                            },
                            "payload_mass_kg": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "payload_inertia_kg_m2": {
                                "type": "object",
                                "required": ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"],
                                "properties": {
                                    "ixx": {"type": "number", "exclusiveMinimum": 0},
                                    "iyy": {"type": "number", "exclusiveMinimum": 0},
                                    "izz": {"type": "number", "exclusiveMinimum": 0},
                                    "ixy": {"type": "number"},
                                    "ixz": {"type": "number"},
                                    "iyz": {"type": "number"},
                                },
                                "additionalProperties": False,
                            },
                            "position_tolerance_m": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "speed_tolerance_mps": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "stable_window_seconds": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "settle_timeout_seconds": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "arguments": {"type": "object"},
                        },
                        "additionalProperties": False,
                    },
                    authority="control",
                    timeout_seconds=20.0,
                )
            ],
        ),
        _definition(
            plugin_id="runtime-actions.ros2-domain-services",
            name="ROS 2 领域动作桥",
            description="将核验、热成像、激光、维修和信标动作路由到具备正向响应的 ROS 2 服务。",
            order=30,
            adapters=[
                _adapter(
                    adapter_id="runtime.domain.ros2-service",
                    executors=[
                        "native.communication.signal-beacon",
                        "native.emergency.verify-drop-zone",
                        "native.inspection.confirm-defect",
                        "native.maintenance.inspect-joint",
                        "native.maintenance.measure-clearance",
                        "native.payload.scan-code",
                        "native.payload.verify-release-area",
                        "native.payload.verify-recipient",
                        "native.sensor.capture-thermal",
                        "native.sensor.lidar-scan",
                        "native.survey.waypoint-sample",
                    ],
                    driver="ros2-service",
                    defaults={
                        "service_namespace": "/dronedream/domain_actions",
                        "service_type": "dronedream_agent_msgs/srv/ExecuteDomainAction",
                    },
                    schema={
                        "type": "object",
                        "required": ["service_name", "service_type", "request"],
                        "properties": {
                            "service_name": {
                                "type": "string",
                                "pattern": "^/(?:[A-Za-z0-9_{}]+/)*[A-Za-z0-9_{}]+$",
                            },
                            "service_type": {
                                "const": "dronedream_agent_msgs/srv/ExecuteDomainAction"
                            },
                            "request": {
                                "type": "object",
                                "required": [
                                    "contract_id",
                                    "task_id",
                                    "action",
                                    "target_node",
                                    "arguments_json",
                                ],
                                "properties": {
                                    "contract_id": {"type": "string", "minLength": 1},
                                    "task_id": {"type": "string", "minLength": 1},
                                    "action": {"type": "string"},
                                    "target_node": {"type": "string", "minLength": 1},
                                    "arguments_json": {"type": "string"},
                                },
                                "additionalProperties": False,
                            },
                        },
                        "additionalProperties": False,
                    },
                    authority="actuate",
                    timeout_seconds=30.0,
                )
            ],
        ),
    ]
    return definitions
