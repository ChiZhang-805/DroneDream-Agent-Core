"""Describe simulation configuration; runtime probes and measured runs prove actual support."""

from __future__ import annotations

import sys
from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin, policy_boolean, policy_number

_PHYSICS_SCHEMA = {
    "type": "object",
    "properties": {
        "solver": {"const": "dart", "default": "dart"},
        "step_seconds": {"type": "number", "exclusiveMinimum": 0, "default": 0.001},
        "real_time_factor_target": {"type": "number", "exclusiveMinimum": 0, "default": 1.0},
        "contact_tolerance_m": {"type": "number", "exclusiveMinimum": 0, "default": 0.0005},
    },
    "additionalProperties": False,
}


# 功能：
#   校验界面选项并描述指定引擎及桥接要求；支持标记必须由运行时探测证实。
# 输入：
#   engine：内置注册表指定的仿真引擎。
#   bridge：该引擎对应的桥接实现名称。
#   configuration：仅允许 headless 布尔选项的配置。
#   _：描述器不使用的扩展参数。
# 输出：
#   descriptor：包含引擎、桥接、界面模式及探测要求的配置描述。
def _descriptor(
    engine: str, bridge: str, *, configuration: dict[str, object], **_: Any
) -> dict[str, object]:
    if not isinstance(configuration, dict) or set(configuration) - {"headless"}:
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    descriptor = {
        "engine": engine,
        "bridge": bridge,
        "headless": policy_boolean(configuration, "headless", True),
        "deterministic_seed_supported": True,
        "runtime_probe_required": True,
    }
    return descriptor


# 功能：
#   验证 DART 参数名称、类型和正值范围；参数语法合格不代表物理仿真已经验收。
# 输入：
#   configuration：求解器、步长、实时因子及接触容差的可选配置。
#   _：物理描述器不使用的扩展参数。
# 输出：
#   result：补齐默认值的独立物理配置。
def _physics(*, configuration: dict[str, object], **_: Any) -> dict[str, object]:
    if (
        not isinstance(configuration, dict)
        or configuration.get("solver", "dart") != "dart"
        or set(configuration) - set(_PHYSICS_SCHEMA["properties"])
    ):
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    result: dict[str, object] = {"solver": "dart"}
    for key, default in (
        ("step_seconds", 0.001),
        ("real_time_factor_target", 1.0),
        ("contact_tolerance_m", 0.0005),
    ):
        number = policy_number(configuration, key, default, 0, sys.float_info.max)
        if number <= 0:
            raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
        result[key] = number
    return result


# 功能：
#   声明需绑定种子的传感器噪声类别，不在此处生成噪声或声称已作用到传感器。
# 输入：
#   _：固定噪声目录不使用的扩展参数。
# 输出：
#   descriptor：惯性、GNSS、气压及相机时延的噪声需求。
def _sensor_noise(**_: Any) -> dict[str, object]:
    descriptor = {
        "models": ["imu-bias-random-walk", "gnss-noise", "barometer-drift", "camera-latency"],
        "seed_bound": True,
    }
    return descriptor


# 功能：
#   声明应由场景统一管理的天气与光照因素，不修改正在运行的仿真世界。
# 输入：
#   _：固定环境目录不使用的扩展参数。
# 输出：
#   descriptor：风雨雾、光照及太阳角度的场景需求。
def _weather_light(**_: Any) -> dict[str, object]:
    descriptor = {
        "models": ["wind", "rain", "fog", "illumination", "sun-angle"],
        "scenario_bound": True,
    }
    return descriptor


# 功能：
#   声明动态场景类别；真实几何、运动轨迹及传感器观测必须来自运行时。
# 输入：
#   _：固定动态场景目录不使用的扩展参数。
# 输出：
#   descriptor：人群、交通、门及电梯的场景需求。
def _crowd_traffic(**_: Any) -> dict[str, object]:
    descriptor = {
        "models": ["pedestrian-flow", "vehicle-traffic", "door-state", "elevator-state"],
        "dynamic_obstacles": True,
    }
    return descriptor


# 功能：
#   请求随仿真暂停的时钟及偏差限制，不把十毫秒要求冒称为已测得的保证。
# 输入：
#   _：固定时钟策略不使用的扩展参数。
# 输出：
#   policy：仿真时间源、暂停感知及最大偏差策略。
def _sim_clock(**_: Any) -> dict[str, object]:
    policy = {"source": "sim-time", "pause_aware": True, "maximum_skew_ms": 10}
    return policy


# 功能：
#   声明多次测试及失败复现要求；仅固定种子不能证明整个引擎执行结果确定。
# 输入：
#   _：固定采样策略不使用的扩展参数。
# 输出：
#   policy：种子策略、最少运行次数及需要汇总的指标名称。
def _monte_carlo(**_: Any) -> dict[str, object]:
    policy = {
        "seed_strategy": "prime-sequence",
        "minimum_runs": 3,
        "failure_reproduction": True,
        "aggregate": ["success-rate", "minimum-clearance", "energy", "latency"],
    }
    return policy


# 功能：
#   注册仿真引擎、环境及采样策略的描述钩子，执行和资格确认仍使用独立门控。
# 输入：
#   无。
# 输出：
#   definitions：包含互斥引擎及可组合环境需求的插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions: list[PluginDefinition] = []
    for order, (plugin_id, name, engine, bridge, enabled) in enumerate(
        [
            ("simulation.engine-gazebo", "Gazebo Harmonic", "gazebo-harmonic", "ros-gz", True),
            ("simulation.engine-isaac", "NVIDIA Isaac Sim", "isaac-sim", "omni-ros2", False),
            ("simulation.engine-airsim", "AirSim", "airsim", "airsim-ros2", False),
            ("simulation.engine-webots", "Webots", "webots", "webots-ros2", False),
        ],
        start=1,
    ):
        # 功能：
        #   冻结本轮引擎与桥接名称，避免循环闭包把所有描述器指向最后一个引擎。
        # 输入：
        #   configuration：当前插件的界面模式配置。
        #   _engine：在定义时绑定的引擎名。
        #   _bridge：在定义时绑定的桥接名。
        #   kwargs：转交通用描述器的扩展参数。
        # 输出：
        #   descriptor：当前引擎的配置描述。
        def handler(
            *,
            configuration: dict[str, object],
            _engine: str = engine,
            _bridge: str = bridge,
            **kwargs: Any,
        ) -> dict[str, object]:
            descriptor = _descriptor(_engine, _bridge, configuration=configuration, **kwargs)
            return descriptor

        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=f"声明 {name} 的时钟、种子和桥接要求，运行时另行探测验证。",
                capability_id=f"{plugin_id}.describe",
                capability_kind="simulator-adapter",
                capability_name=name,
                capability_description=f"{name} 仿真器适配合同。",
                category_id="simulation",
                category_label="仿真与测试",
                slot_id="simulation.simulator-descriptor",
                slot_label="仿真器",
                activation_mode="single",
                category_order=80,
                slot_order=10,
                plugin_order=order * 10,
                hooks={"describe_simulator": handler},
                default_enabled=enabled,
                failure_mode="fail-closed",
                configuration_schema={
                    "type": "object",
                    "properties": {"headless": {"type": "boolean", "default": True}},
                    "additionalProperties": False,
                },
            )
        )
    multiple = [
        (
            "simulation.physics-dart",
            "DART 物理",
            "physics-model",
            "simulation.physics-models",
            "物理模型",
            "describe_physics",
            _physics,
            10,
        ),
        (
            "simulation.sensor-noise",
            "传感器噪声",
            "sensor-model",
            "simulation.sensor-models",
            "传感器模型",
            "describe_sensor",
            _sensor_noise,
            20,
        ),
        (
            "simulation.weather-light",
            "天气与光照",
            "environment-model",
            "simulation.environment-models",
            "环境模型",
            "describe_environment",
            _weather_light,
            30,
        ),
        (
            "simulation.crowd-traffic",
            "人群与交通",
            "environment-model",
            "simulation.environment-models",
            "环境模型",
            "describe_environment",
            _crowd_traffic,
            40,
        ),
    ]
    for plugin_id, name, kind, slot_id, slot_label, hook, handler, order in multiple:
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=f"为仿真任务提供可冻结的{name}参数。",
                capability_id=f"{plugin_id}.{hook}",
                capability_kind=kind,
                capability_name=name,
                capability_description=f"{name}合同。",
                category_id="simulation",
                category_label="仿真与测试",
                slot_id=slot_id,
                slot_label=slot_label,
                activation_mode="multiple",
                category_order=80,
                slot_order=40,
                plugin_order=order,
                hooks={hook: handler},
                default_enabled=True,
                failure_mode="isolate",
                configuration_schema=_PHYSICS_SCHEMA if handler is _physics else {},
            )
        )
    for plugin_id, name, kind, slot_id, hook, handler, order in [
        (
            "simulation.clock-sim-time",
            "仿真时钟",
            "clock-policy",
            "simulation.clock-policy",
            "resolve_clock",
            _sim_clock,
            10,
        ),
        (
            "simulation.monte-carlo-prime",
            "Monte Carlo 素数种子",
            "monte-carlo-policy",
            "simulation.monte-carlo-policy",
            "resolve_monte_carlo",
            _monte_carlo,
            20,
        ),
    ]:
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=f"提供{name}的可复现策略。",
                capability_id=f"{plugin_id}.{hook}",
                capability_kind=kind,
                capability_name=name,
                capability_description=f"{name}策略。",
                category_id="simulation",
                category_label="仿真与测试",
                slot_id=slot_id,
                slot_label=name,
                activation_mode="single",
                category_order=80,
                slot_order=50,
                plugin_order=order,
                hooks={hook: handler},
                default_enabled=True,
                failure_mode="fail-closed",
            )
        )
    return definitions
