"""Describe domain actions, executors and required receipts; declaring an action does not run it."""

from __future__ import annotations

from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_plugin_sdk.protocol import copy_json

from ._helpers import hook_plugin


# 功能：
#   生成独立的领域动作契约，声明成功证据和执行器标识，不调用执行器或授予动作权限。
# 输入：
#   action_id：领域动作的唯一标识。
#   domain_id：动作归属的领域标识。
#   label：界面使用的动作名称。
#   description：动作意图及边界说明。
#   movement：该动作是否包含移动。
#   payload：该动作是否改变载荷状态。
#   boundary：起飞、着陆或无飞行边界类别。
#   success：必须由实际执行证实的成功条件。
#   fallbacks：动作失败时允许考虑的退路标识。
#   simulator：仿真执行器标识。
#   runtime：可选真机执行器标识。
#   authority：执行前需要独立核准的权限类别。
#   input_schema：动作参数约束；缺省时禁止额外参数。
# 输出：
#   action：不共享可变输入容器的动作定义对象。
def _action(
    action_id: str,
    domain_id: str,
    label: str,
    description: str,
    *,
    movement: bool = False,
    payload: bool = False,
    boundary: str = "none",
    success: list[str],
    fallbacks: list[str],
    simulator: str,
    runtime: str | None = None,
    authority: str = "plan",
    input_schema: dict[str, object] | None = None,
) -> dict[str, object]:
    action = copy_json(
        {
            "schema_version": "dronedream.action-definition.v1",
            "action_id": action_id,
            "domain_id": domain_id,
            "label": label,
            "description": description,
            "movement": movement,
            "payload": payload,
            "flight_boundary": boundary,
            "input_schema": (
                {"type": "object", "additionalProperties": False}
                if input_schema is None
                else input_schema
            ),
            "required_success_evidence": success,
            "allowed_fallbacks": fallbacks,
            "simulator_executor": simulator,
            "runtime_executor": runtime,
            "authority": authority,
        }
    )
    return action


# 功能：
#   冻结动作包内容，使随后修改源列表或某次调用结果不能改变其他任务看到的定义。
# 输入：
#   domain_id：动作包的领域标识。
#   actions：本包提供的动作定义列表。
# 输出：
#   declare：返回独立动作包内容的声明回调。
def _pack(domain_id: str, actions: list[dict[str, object]]):
    frozen = copy_json(actions)

    # 功能：
    #   返回本次任务可读取的动作定义，执行器可用性与准入仍由运行层单独检查。
    # 输入：
    #   _：固定声明不使用的扩展调用参数。
    # 输出：
    #   pack：含领域标识及独立动作列表的对象。
    def declare(**_: Any) -> dict[str, object]:
        pack = {"domain_id": domain_id, "actions": copy_json(frozen)}
        return pack

    return declare


# 功能：
#   将领域动作包注册为可组合插件，冻结声明内容并要求仅在下一任务切换。
# 输入：
#   plugin_id：插件标识。
#   name：插件显示名称。
#   description：插件用途描述。
#   domain_id：包内动作的领域标识。
#   actions：动作契约列表。
#   order：同插槽内的显示与装配顺序。
#   failure_mode：动作包声明失败时的处理方式。
# 输出：
#   definition：包含动作声明回调及元数据的插件定义。
def _definition(
    *,
    plugin_id: str,
    name: str,
    description: str,
    domain_id: str,
    actions: list[dict[str, object]],
    order: int,
    failure_mode: str = "fail-closed",
) -> PluginDefinition:
    definition = hook_plugin(
        module_name=__name__,
        plugin_id=plugin_id,
        name=name,
        description=description,
        capability_id=f"{plugin_id}.actions",
        capability_kind="action-pack",
        capability_name=name,
        capability_description=description,
        category_id="mission-domain",
        category_label="任务领域与动作",
        slot_id="mission.action-packs",
        slot_label="任务动作包",
        activation_mode="multiple",
        category_order=25,
        slot_order=10,
        plugin_order=order,
        hooks={"declare_actions": _pack(domain_id, actions)},
        default_enabled=True,
        failure_mode=failure_mode,
        swap_policy="next-mission",
        metadata={"domain_id": domain_id, "action_ids": [item["action_id"] for item in actions]},
    )
    return definition


# 功能：
#   1. 声明飞行、配送、巡检、测绘、应急和维护动作，声明不等于硬件已具备能力。
#   2. 将接触前稳定、挂载物理交接和挂载后稳定分为独立要求，禁止互相冒充成功证明。
# 输入：
#   无。
# 输出：
#   definitions：本产品内置的领域动作包定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    core = "core.flight"
    delivery = "delivery.custody"
    inspection = "inspection.infrastructure"
    survey = "survey.mapping"
    emergency = "emergency.response"
    maintenance = "maintenance.facility"
    definitions = [
        _definition(
            plugin_id="actions.core-flight",
            name="核心飞行动作",
            description="保留起飞、移动、返航和降落边界，是所有任务动作包的基础。",
            domain_id=core,
            order=10,
            actions=[
                _action(
                    "none",
                    core,
                    "无载荷动作",
                    "明确表示任务不需要载荷操作。",
                    success=["no payload action required"],
                    fallbacks=["continue", "hold"],
                    simulator="sim.core.noop",
                ),
                _action(
                    "takeoff",
                    core,
                    "起飞",
                    "从已验证起飞点进入稳定悬停。",
                    boundary="takeoff",
                    success=["airborne", "stable hover"],
                    fallbacks=["land", "abort"],
                    simulator="sim.flight.takeoff",
                    runtime="native.flight.takeoff",
                    authority="control",
                ),
                _action(
                    "traverse",
                    core,
                    "通行",
                    "沿已验证路线穿越建筑或室外连接区域。",
                    movement=True,
                    success=["segment complete", "clearance maintained"],
                    fallbacks=["hold", "return", "land"],
                    simulator="sim.flight.follow-track",
                    runtime="native.flight.follow-track",
                    authority="control",
                ),
                _action(
                    "navigate",
                    core,
                    "导航",
                    "导航至合同中授权的目标节点。",
                    movement=True,
                    success=["target reached", "pose stable"],
                    fallbacks=["hold", "return", "land"],
                    simulator="sim.flight.follow-track",
                    runtime="native.flight.follow-track",
                    authority="control",
                ),
                _action(
                    "return",
                    core,
                    "返程",
                    "返回合同中冻结的返程节点。",
                    movement=True,
                    success=["return node reached"],
                    fallbacks=["hold", "land", "abort"],
                    simulator="sim.flight.follow-track",
                    runtime="native.flight.follow-track",
                    authority="control",
                ),
                _action(
                    "land",
                    core,
                    "降落",
                    "在已验证着陆区域完成受控降落。",
                    boundary="landing",
                    success=["landed", "motors safe"],
                    fallbacks=["hold", "abort"],
                    simulator="sim.flight.land",
                    runtime="native.flight.land",
                    authority="control",
                ),
            ],
        ),
        _definition(
            plugin_id="actions.delivery-custody",
            name="配送与交接动作",
            description="提供到点悬停后的挂载、载荷交接与释放动作，不要求扫码或收件人验证。",
            domain_id=delivery,
            order=20,
            actions=[
                _action(
                    "pickup",
                    delivery,
                    "取件",
                    "在已授权取件点稳定悬停后完成抓取或挂载，并读取实际挂载状态。",
                    payload=True,
                    success=["payload attached", "attachment state readback"],
                    fallbacks=["hold", "return", "abort"],
                    simulator="sim.payload.pickup",
                    runtime="native.payload.pickup",
                    authority="actuate",
                ),
                _action(
                    "delivery.release-payload",
                    delivery,
                    "释放载荷",
                    "在接收方和放置区域均确认后释放载荷。",
                    payload=True,
                    success=["payload released", "release verified"],
                    fallbacks=["hold", "return", "abort"],
                    simulator="sim.payload.release",
                    runtime="native.payload.release",
                    authority="actuate",
                ),
                _action(
                    "delivery.verify-release-area",
                    delivery,
                    "确认释放区域",
                    "在释放载荷前确认放置区域无人员、障碍物或其他禁止条件。",
                    success=["release area clear"],
                    fallbacks=["hold", "return", "abort"],
                    simulator="sim.payload.verify-release-area",
                    runtime="native.payload.verify-release-area",
                ),
            ],
        ),
        _definition(
            plugin_id="actions.payload-transition-guards",
            name="载荷转换守卫",
            description=(
                "把接触前稳定证明与挂载后的物理交接证明拆成独立动作，"
                "避免用扫码结果冒充飞行稳定或载荷动力学证据。"
            ),
            domain_id="delivery.transition",
            order=25,
            actions=[
                _action(
                    "delivery.precontact-hold",
                    "delivery.transition",
                    "接触前稳定悬停",
                    "在接触载荷前证明位置和速度稳定，并确认载荷仍处于分离状态。",
                    success=["stable pre-contact hover", "no payload contact"],
                    fallbacks=["hold", "return", "abort"],
                    simulator="sim.payload.precontact-hold",
                    runtime="native.payload.precontact-hold",
                    authority="control",
                ),
                _action(
                    "delivery.confirm-custody",
                    "delivery.transition",
                    "确认载荷交接",
                    "挂载后验证连接状态及哈希绑定的载荷质量与惯量，建立可信交接状态。",
                    success=[
                        "payload attachment confirmed",
                        "mass and inertia update confirmed",
                        "custody state accepted",
                    ],
                    fallbacks=["hold", "return", "abort"],
                    simulator="sim.payload.confirm-custody",
                    runtime="native.payload.confirm-custody",
                    authority="control",
                ),
            ],
        ),
        _definition(
            plugin_id="actions.loaded-flight-stability",
            name="挂载后飞行稳定性",
            description=(
                "在载荷物理交接之后、返程之前，以独立稳定窗口证明飞行器能承受新增质量；"
                "它不重复关节状态确认，也不负责路径规划。"
            ),
            domain_id="delivery.transition",
            order=27,
            actions=[
                _action(
                    "delivery.verify-loaded-stability",
                    "delivery.transition",
                    "验证载荷悬停稳定",
                    "保持取货航点并验证挂载后的位置误差、速度和持续稳定时间，再授权返程。",
                    success=[
                        "loaded hover stable",
                        "post-attachment dynamics accepted",
                        "return authorized",
                    ],
                    fallbacks=["hold", "land", "abort"],
                    simulator="sim.payload.verify-loaded-stability",
                    runtime="native.payload.verify-loaded-stability",
                    authority="control",
                )
            ],
        ),
        _definition(
            plugin_id="actions.infrastructure-inspection",
            name="设施巡检动作",
            description="提供稳定视点、RGB、热成像和缺陷复核动作。",
            domain_id=inspection,
            order=30,
            actions=[
                _action(
                    "inspection.capture-rgb",
                    inspection,
                    "采集可见光图像",
                    "在稳定视点采集带位姿和时间戳的 RGB 证据。",
                    success=["image captured", "pose bound", "timestamp bound"],
                    fallbacks=["hold", "continue", "return"],
                    simulator="sim.sensor.capture-rgb",
                    runtime="native.sensor.capture-rgb",
                ),
                _action(
                    "inspection.capture-thermal",
                    inspection,
                    "采集热成像",
                    "采集带温度标定和位姿绑定的热成像证据。",
                    success=["thermal frame captured", "calibration valid"],
                    fallbacks=["hold", "continue", "return"],
                    simulator="sim.sensor.capture-thermal",
                    runtime="native.sensor.capture-thermal",
                ),
                _action(
                    "inspection.facade-pass",
                    inspection,
                    "立面巡检航段",
                    "保持安全净空和成像重叠率完成立面航段。",
                    movement=True,
                    success=["coverage complete", "clearance maintained"],
                    fallbacks=["hold", "return", "land"],
                    simulator="sim.inspection.facade-pass",
                    runtime="native.inspection.facade-pass",
                    authority="control",
                ),
                _action(
                    "inspection.confirm-defect",
                    inspection,
                    "缺陷复核",
                    "从第二视角重新采集疑似缺陷并绑定证据。",
                    success=["second viewpoint captured", "defect record bound"],
                    fallbacks=["hold", "continue", "return"],
                    simulator="sim.inspection.confirm-defect",
                    runtime="native.inspection.confirm-defect",
                ),
            ],
        ),
        _definition(
            plugin_id="actions.area-survey",
            name="区域测绘动作",
            description="提供网格影像、激光扫描和采样点任务。",
            domain_id=survey,
            order=40,
            actions=[
                _action(
                    "survey.grid-capture",
                    survey,
                    "网格影像采集",
                    "按照重叠率和地面分辨率要求执行网格覆盖。",
                    movement=True,
                    success=["coverage target met", "image overlap met"],
                    fallbacks=["hold", "return", "land"],
                    simulator="sim.survey.grid-capture",
                    runtime="native.survey.grid-capture",
                    authority="control",
                ),
                _action(
                    "survey.lidar-scan",
                    survey,
                    "激光点云扫描",
                    "采集带时间同步和位姿绑定的点云。",
                    success=["point cloud captured", "time sync valid"],
                    fallbacks=["hold", "continue", "return"],
                    simulator="sim.sensor.lidar-scan",
                    runtime="native.sensor.lidar-scan",
                ),
                _action(
                    "survey.waypoint-sample",
                    survey,
                    "航点采样",
                    "在合同指定采样点完成传感器读数采集。",
                    success=["sample captured", "location bound"],
                    fallbacks=["hold", "continue", "return"],
                    simulator="sim.survey.waypoint-sample",
                    runtime="native.survey.waypoint-sample",
                ),
            ],
        ),
        _definition(
            plugin_id="actions.emergency-response",
            name="应急响应动作",
            description="提供扇区搜索、医疗包投放和信标动作，安全授权保持核心所有。",
            domain_id=emergency,
            order=50,
            actions=[
                _action(
                    "emergency.search-sector",
                    emergency,
                    "扇区搜索",
                    "在给定时间和能源预算内搜索指定区域。",
                    movement=True,
                    success=["sector coverage complete", "detections recorded"],
                    fallbacks=["hold", "return", "land"],
                    simulator="sim.emergency.search-sector",
                    runtime="native.emergency.search-sector",
                    authority="control",
                ),
                _action(
                    "emergency.drop-kit",
                    emergency,
                    "投放应急物资",
                    "确认投放区域无人后释放医疗或通信物资。",
                    payload=True,
                    success=["payload released", "release verified"],
                    fallbacks=["hold", "return", "abort"],
                    simulator="sim.payload.drop-kit",
                    runtime="native.payload.release",
                    authority="actuate",
                ),
                _action(
                    "emergency.verify-drop-zone",
                    emergency,
                    "确认投放区域",
                    "在应急物资释放前确认投放区域内没有人员或障碍物。",
                    success=["drop zone clear"],
                    fallbacks=["hold", "return", "abort"],
                    simulator="sim.emergency.verify-drop-zone",
                    runtime="native.emergency.verify-drop-zone",
                ),
                _action(
                    "emergency.signal-beacon",
                    emergency,
                    "发送定位信标",
                    "在任务目标附近发送带位置的应急信标。",
                    success=["beacon transmitted", "location acknowledged"],
                    fallbacks=["continue", "return"],
                    simulator="sim.communication.signal-beacon",
                    runtime="native.communication.signal-beacon",
                ),
            ],
        ),
        _definition(
            plugin_id="actions.facility-maintenance",
            name="设施维护动作",
            description="提供连接处复核、净空测量和材料状态检查动作。",
            domain_id=maintenance,
            order=60,
            actions=[
                _action(
                    "maintenance.inspect-joint",
                    maintenance,
                    "检查连接处",
                    "从多个角度检查楼体、道路或设备连接处的空隙与穿插。",
                    success=["multi-angle evidence", "joint classification recorded"],
                    fallbacks=["hold", "continue", "return"],
                    simulator="sim.maintenance.inspect-joint",
                    runtime="native.maintenance.inspect-joint",
                ),
                _action(
                    "maintenance.measure-clearance",
                    maintenance,
                    "测量净空",
                    "测量目标连接处或通行区域的最小净空。",
                    success=["clearance measured", "measurement pose bound"],
                    fallbacks=["hold", "continue", "return"],
                    simulator="sim.maintenance.measure-clearance",
                    runtime="native.maintenance.measure-clearance",
                ),
            ],
        ),
    ]
    return definitions
