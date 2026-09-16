"""Native capability bindings and bounded policies, not Python flight implementations."""

from __future__ import annotations

from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin


# 功能：
#   描述需要原生宿主解析和授权的 ABI 绑定；声明 EKF/MPC 不等于已经安装、估计或执行。
# 输入：
#   name：原生实现名称。
#   capabilities：该实现宣告支持的能力名称。
#   _：描述器不使用的扩展参数。
# 输出：
#   descriptor：独立的能力列表、ABI 标识及核心授权要求。
def _descriptor(name: str, capabilities: list[str], **_: Any) -> dict[str, object]:
    descriptor = {
        "implementation": name,
        "capabilities": list(capabilities),
        "native_abi": "dronedream.capability-plugin.v1",
        "core_authorization_required": True,
    }
    return descriptor


# 功能：
#   1. 校验原生进程存活检测的整数毫秒参数及心跳与截止时间关系，不接受静默类型转换。
#   2. 生成启动、调度容忍及持续失联处置策略，不在此处监测进程或发送降落指令。
# 输入：
#   configuration：可选时间参数组成的字典。
#   _：策略解析器不使用的扩展参数。
# 输出：
#   policy：补齐默认值并限制范围的失联处置策略。
def _watchdog(*, configuration: dict[str, object], **_: Any) -> dict[str, object]:
    limits = {
        "deadline_ms": (1_000, 20, 1_000),
        "startup_deadline_ms": (30_000, 1_000, 60_000),
        "heartbeat_ms": (25, 5, 250),
        "scheduling_jitter_grace_ms": (250, 0, 500),
        "persistent_miss_ms": (250, 25, 1_000),
    }
    if not isinstance(configuration, dict) or configuration.keys() - limits.keys():
        raise ValueError("NATIVE_WATCHDOG_CONFIGURATION_INVALID")
    resolved: dict[str, int] = {}
    for key, (default, minimum, maximum) in limits.items():
        value = configuration.get(key, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("NATIVE_WATCHDOG_CONFIGURATION_INVALID")
        resolved[key] = value
    if resolved["heartbeat_ms"] > resolved["deadline_ms"]:
        raise ValueError("NATIVE_WATCHDOG_HEARTBEAT_EXCEEDS_DEADLINE")
    policy = {
        # Windows/WSL 调度不是硬实时：此上限只约束进程响应，不能让一秒前的观测自动有效。
        # 感知新鲜度、制动距离仍须独立核验，真机需重新测量并认证相应时限。
        "deadline_ms": resolved["deadline_ms"],
        # 启动等待发生在飞行授权前；执行器进入活动阶段后才适用飞行期截止时间。
        "startup_deadline_ms": resolved["startup_deadline_ms"],
        "heartbeat_ms": resolved["heartbeat_ms"],
        # 显式限制调度抖动宽限，防止一次桌面调度停顿被直接解释为传感器永久失联。
        "scheduling_jitter_grace_ms": resolved["scheduling_jitter_grace_ms"],
        # 持续失联时间用于区分短暂抖动与需要升级处置的故障；具体动作由原生宿主执行。
        "persistent_miss_ms": resolved["persistent_miss_ms"],
        "on_miss": "safe-hold-then-land",
        "fail_closed": True,
    }
    return policy


# 功能：
#   注册传输、状态估计、控制和遥测等原生绑定插槽，认证更新不能直接热替换飞行权限。
# 输入：
#   无。
# 输出：
#   definitions：原生能力发现和存活检测策略的插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions: list[PluginDefinition] = []
    # 这些包装只返回绑定元数据；原生库是否存在及其资格不能从名称或默认启用推断。
    singles = [
        (
            "native.transport-px4",
            "PX4 uXRCE-DDS",
            "transport",
            "native.transport",
            "飞控传输",
            "transport_message",
            lambda **kwargs: _descriptor(
                "px4-uxrce-dds", ["offboard", "vehicle-command", "timesync"], **kwargs
            ),
            True,
            10,
        ),
        (
            "native.transport-ardupilot",
            "ArduPilot DDS/MAVLink",
            "transport",
            "native.transport",
            "飞控传输",
            "transport_message",
            lambda **kwargs: _descriptor(
                "ardupilot-dds-mavlink", ["guided", "mission", "timesync"], **kwargs
            ),
            False,
            20,
        ),
        (
            "native.estimator-ekf2",
            "EKF2 状态估计",
            "state-estimator",
            "native.state-estimator",
            "状态估计",
            "estimate_state",
            lambda **kwargs: _descriptor(
                "ekf2", ["imu", "gnss", "barometer", "magnetometer"], **kwargs
            ),
            True,
            10,
        ),
        (
            "native.estimator-factor-graph",
            "因子图状态估计",
            "state-estimator",
            "native.state-estimator",
            "状态估计",
            "estimate_state",
            lambda **kwargs: _descriptor("factor-graph", ["imu", "vio", "uwb", "lidar"], **kwargs),
            False,
            20,
        ),
        (
            "native.localization-gnss-vio-slam",
            "GNSS/VIO/SLAM 融合定位",
            "localization",
            "native.localization",
            "定位",
            "localize",
            lambda **kwargs: _descriptor(
                "gnss-vio-slam", ["gnss", "vio", "slam", "uwb", "frame-transform"], **kwargs
            ),
            True,
            10,
        ),
        (
            "native.controller-mpc",
            "约束 MPC 控制器",
            "controller",
            "native.controller",
            "飞行控制器",
            "control_policy",
            lambda **kwargs: _descriptor(
                "bounded-mpc", ["position", "velocity", "landing", "collision-brake"], **kwargs
            ),
            True,
            10,
        ),
    ]
    for plugin_id, name, kind, slot_id, slot_label, hook, handler, enabled, order in singles:
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=f"通过原生类型 ABI 提供{name}，运行期变更需认证安装。",
                capability_id=f"{plugin_id}.{hook}",
                capability_kind=kind,
                capability_name=name,
                capability_description=f"{name}原生运行时合同。",
                category_id="native",
                category_label="ROS 2 与飞控",
                slot_id=slot_id,
                slot_label=slot_label,
                activation_mode="single",
                category_order=75,
                slot_order=order,
                plugin_order=order,
                hooks={hook: handler},
                default_enabled=enabled,
                failure_mode="fail-closed",
                swap_policy="certified-update",
                permissions=["ros.read", "ros.write", "telemetry.read"],
            )
        )
    multiples = [
        (
            "native.telemetry-core",
            "核心遥测",
            "telemetry-adapter",
            "native.telemetry",
            "遥测",
            "normalize_telemetry",
            ["pose", "velocity", "battery", "link", "flight-mode"],
            10,
        ),
        (
            "native.perception-obstacles",
            "障碍感知",
            "perception",
            "native.perception",
            "感知",
            "normalize_telemetry",
            ["depth", "lidar", "occupancy", "dynamic-track"],
            20,
        ),
        (
            "native.payload-gripper",
            "抓取与挂载",
            "payload-driver",
            "native.payload-drivers",
            "载荷驱动",
            "payload_command",
            ["attach", "detach", "load-cell", "identity"],
            30,
        ),
        (
            "native.payload-gimbal",
            "云台与相机",
            "payload-driver",
            "native.payload-drivers",
            "载荷驱动",
            "payload_command",
            ["gimbal", "photo", "video", "focus"],
            40,
        ),
        (
            "native.blackbox-rosbag-ulog",
            "ROS Bag 与 ULog 黑匣子",
            "evidence",
            "native.blackbox",
            "黑匣子",
            "normalize_telemetry",
            ["rosbag2", "mcap", "ulog", "hash-chain"],
            50,
        ),
    ]
    for plugin_id, name, kind, slot_id, slot_label, hook, capabilities, order in multiples:
        # 功能：
        #   在定义时捕获当前插件身份和能力列表，调用时返回独立的发现元数据。
        # 输入：
        #   _name：当前循环绑定的原生插件名。
        #   _capabilities：当前循环绑定的能力名称列表。
        #   kwargs：转交描述器的扩展参数。
        # 输出：
        #   descriptor：当前原生插件的能力描述。
        def handler(
            *, _name: str = plugin_id, _capabilities: list[str] = capabilities, **kwargs: Any
        ) -> dict[str, object]:
            descriptor = _descriptor(_name, _capabilities, **kwargs)
            return descriptor

        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=f"可组合的{name}原生能力。",
                capability_id=f"{plugin_id}.{hook}",
                capability_kind=kind,
                capability_name=name,
                capability_description=f"{name}原生能力合同。",
                category_id="native",
                category_label="ROS 2 与飞控",
                slot_id=slot_id,
                slot_label=slot_label,
                activation_mode="multiple",
                category_order=75,
                slot_order=50,
                plugin_order=order,
                hooks={hook: handler},
                default_enabled=True,
                failure_mode="isolate",
                swap_policy="certified-update",
                permissions=["ros.read", "telemetry.read", "evidence.write"],
            )
        )
    definitions.append(
        hook_plugin(
            module_name=__name__,
            plugin_id="native.watchdog-deadline",
            name="实时截止时间看门狗",
            description="监控原生插件健康与截止时间，失约时强制安全悬停并降落。",
            capability_id="native.watchdog-deadline.resolve",
            capability_kind="runtime-watchdog",
            capability_name="实时截止时间看门狗",
            capability_description="原生运行时失效关闭策略。",
            category_id="native",
            category_label="ROS 2 与飞控",
            slot_id="native.watchdog",
            slot_label="实时看门狗",
            activation_mode="single",
            category_order=75,
            slot_order=60,
            plugin_order=10,
            hooks={"resolve_watchdog": _watchdog},
            default_enabled=True,
            failure_mode="fail-closed",
            swap_policy="certified-update",
            configuration_schema={
                "type": "object",
                "properties": {
                    "deadline_ms": {"type": "integer", "minimum": 20, "maximum": 1000},
                    "startup_deadline_ms": {
                        "type": "integer",
                        "minimum": 1000,
                        "maximum": 60000,
                    },
                    "heartbeat_ms": {"type": "integer", "minimum": 5, "maximum": 250},
                    "scheduling_jitter_grace_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 500,
                    },
                    "persistent_miss_ms": {
                        "type": "integer",
                        "minimum": 25,
                        "maximum": 1000,
                    },
                },
                "additionalProperties": False,
            },
            permissions=["ros.read", "ros.write", "telemetry.read"],
        )
    )
    return definitions
