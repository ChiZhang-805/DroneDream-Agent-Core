"""Resolve every supported map input into one typed runtime-binding contract."""

from __future__ import annotations

from typing import Any

from dronedream_plugin_sdk.protocol import encode_json

from .contracts import MapRuntimeBindings, Vector3, VehicleAsset


class MapRuntimeBindingsError(ValueError):
    """Stable qualification/runtime failure for incomplete map bindings."""


# 功能：
#   仅从所选地图的当前语义结构读取运行绑定，拒绝旧结构、隐式类型转换和内置地图补值。
# 输入：
#   semantic：当前地图的语义字典。
# 输出：
#   bindings：严格校验且与输入容器分离的仿真坐标与启动绑定。
def load_map_runtime_bindings(semantic: dict[str, Any]) -> MapRuntimeBindings:
    if type(semantic) is not dict:
        raise MapRuntimeBindingsError("MAP_SEMANTIC_INVALID")
    if semantic.get("schema_version") != "dronedream.map-semantic.v1":
        raise MapRuntimeBindingsError("MAP_SEMANTIC_SCHEMA_OBSOLETE")
    # A visual/semantic asset is not executable until its own runtime bindings
    # validate. Never fill gaps from whichever built-in map happens to be loaded.
    normalized = semantic.get("runtime_bindings")
    if not isinstance(normalized, dict):
        raise MapRuntimeBindingsError("MAP_RUNTIME_BINDINGS_MISSING")
    try:
        bindings = MapRuntimeBindings.model_validate_json(encode_json(normalized), strict=True)
    except ValueError as error:
        raise MapRuntimeBindingsError("MAP_RUNTIME_BINDINGS_INVALID") from error
    return bindings


# 功能：
#   重新核验所选飞机的实际碰撞偏移与机身包络，不猜测零偏移，也不返回可改写原资产的引用。
# 输入：
#   vehicle：当前所选飞机资产，可能在构造后被其他调用方改变。
# 输出：
#   offset：独立的模型坐标系碰撞中心偏移，单位米。
def resolve_vehicle_collision_center_offset(vehicle: VehicleAsset) -> Vector3:
    if not isinstance(vehicle, VehicleAsset):
        raise MapRuntimeBindingsError("VEHICLE_COLLISION_CENTER_OFFSET_INVALID")
    try:
        vehicle = VehicleAsset.model_validate_json(
            encode_json(vehicle.model_dump(mode="json")), strict=True
        )
    except ValueError as error:
        raise MapRuntimeBindingsError("VEHICLE_COLLISION_CENTER_OFFSET_INVALID") from error
    if vehicle.collision_center_offset_model_m is not None:
        offset = vehicle.collision_center_offset_model_m
        return offset
    # Guessing zero here would move the collision volume without moving the
    # selected vehicle, invalidating its clearance evidence.
    raise MapRuntimeBindingsError("VEHICLE_COLLISION_CENTER_OFFSET_MISSING")
