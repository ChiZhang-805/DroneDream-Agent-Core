"""Shared task, sensor and vehicle context for deployed and teacher observations."""

import math
from collections.abc import Mapping

from .contracts import VehicleAsset

_STRATEGIC_CONTEXT_SECTIONS = frozenset({"task", "map", "sensor_contract", "vehicle", "payload"})


# 功能：
#   1. 复制有限、有界的战略描述，保持字段身份及零、缺失与布尔语义。
#   2. 只正规化描述文字，不截断字段名，不将描述视为执行指令或传感器健康证据。
# 输入：
#   context：五个允许板块组成的上下文；None 表示没有补充描述。
# 输出：
#   result：与原始可变容器隔离的有限 JSON 字典。
def bounded_navigation_context(context: Mapping[str, object] | None) -> dict[str, object]:
    if context is None:
        result = {}
        return result
    if not isinstance(context, Mapping) or len(context) > len(_STRATEGIC_CONTEXT_SECTIONS):
        raise ValueError("strategic navigation context must be a bounded mapping")
    unknown = set(context) - _STRATEGIC_CONTEXT_SECTIONS
    if unknown:
        raise ValueError("strategic navigation context has unsupported sections")
    remaining = 4096

    # 功能：
    #   递归复制单项描述，每次访问扣减共享节点预算，避免合法局部结构组合成巨大工作量。
    # 输入：
    #   value：当前描述项。
    #   depth：当前嵌套层级。
    # 输出：
    #   result：保留数值语义并完成独立复制的描述项。
    def bounded(value: object, *, depth: int) -> object:
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise ValueError("strategic navigation context exceeds the node budget")
        if depth > 3:
            raise ValueError("strategic navigation context is too deeply nested")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("strategic navigation context contains a non-finite number")
        if isinstance(value, int) and value.bit_length() > 64:
            raise ValueError("strategic navigation context integer exceeds the field budget")
        if value is None or isinstance(value, (bool, int, float)):
            result = value
            return result
        if isinstance(value, str):
            # 先限制原文，再 split；输出截断本身不能限制处理巨大原文的成本。
            if len(value) > 4096:
                raise ValueError("strategic navigation context text exceeds the input budget")
            clean = " ".join(value.split())
            if not clean:
                raise ValueError("strategic navigation context contains an empty string")
            result = clean[:160]
            return result
        if isinstance(value, Mapping):
            if len(value) > 24:
                raise ValueError("strategic navigation context mapping is too large")
            if any(not isinstance(key, str) or not key or len(key) > 64 for key in value):
                raise ValueError("strategic navigation context has an invalid field name")
            result = {
                key: bounded(item, depth=depth + 1)
                for key, item in sorted(value.items())
            }
            return result
        if isinstance(value, (list, tuple)):
            if len(value) > 32:
                raise ValueError("strategic navigation context list is too large")
            result = [bounded(item, depth=depth + 1) for item in value]
            return result
        raise ValueError("strategic navigation context contains an unsupported value")

    result = {section: bounded(context[section], depth=0) for section in sorted(context)}
    return result


# 功能：
#   1. 组装任务、地图、机体和传感器配置描述，拒绝补充字段覆盖固定控制边界。
#   2. 配置名称不代表实时可用；实际运动许可仍依赖独立的健康、掩码和来源时钟。
# 输入：
#   task：当前任务阶段描述。
#   map_context：地图与语义区域描述。
#   sensor_context：注册表身份、健康状态等补充信息。
#   vehicle：当前机体物理配置。
#   payload：当前负载状态描述。
#   rgb_enabled：本次配置是否启用前向 RGB。
# 输出：
#   result：经过边界检查且不再引用调用方容器的战略上下文。
def build_navigation_context(*, task: dict, map_context: dict, sensor_context: dict,
                             vehicle: VehicleAsset, payload: dict, rgb_enabled: bool) -> dict:
    if type(rgb_enabled) is not bool or not isinstance(vehicle, VehicleAsset):
        raise ValueError("navigation context requires a vehicle and explicit RGB boolean")
    if any(type(value) is not dict for value in (task, map_context, sensor_context, payload)):
        raise ValueError("navigation context sections must be objects")
    sensor_contract = {
        "metric_authority": "calibrated depth plus localization",
        "rgb_role": "supplementary semantic evidence" if rgb_enabled else "not enabled",
        "unknown_space_policy": "blocked",
        "coordinate_frame": "deployment map ENU collision-envelope center",
        "configured_vehicle_sensors": vehicle.sensors,
        "active_runtime_sensors": [
            "oakd-lite-depth", "px4-native-estimator", "px4-identity-telemetry",
            *(["px4-dynamics-telemetry"] if "dynamics" in payload else []),
            *(["forward-rgb"] if rgb_enabled else []),
        ],
    }
    for key in sensor_contract.keys() & sensor_context.keys():
        expected, supplied = sensor_contract[key], sensor_context[key]
        if type(supplied) is not type(expected) or supplied != expected:
            raise ValueError("sensor context cannot override the configured authority contract")
    result = bounded_navigation_context({
        "task": task,
        "map": map_context,
        "sensor_contract": {**sensor_contract, **sensor_context},
        "vehicle": {
            "asset_id": vehicle.asset_id,
            "dry_mass_kg": vehicle.dry_mass_kg,
            "maximum_takeoff_mass_kg": vehicle.max_takeoff_mass_kg,
            "maximum_speed_mps": vehicle.max_speed_mps,
            "maximum_acceleration_mps2": vehicle.max_acceleration_mps2,
            "body_radius_m": vehicle.body_radius_m,
            "body_height_m": vehicle.body_height_m,
        },
        "payload": payload,
    })
    return result
