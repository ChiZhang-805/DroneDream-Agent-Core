"""Resolve every supported map input into one typed runtime-binding contract."""

from __future__ import annotations

import math
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


# 功能：
#   1. 从地图在候选出生点下方的真实水平支撑面与机体碰撞包络，计算落地时的模型根位姿。
#   2. 将地图声明的出生点解释为碰撞中心的水平位置，不把地图通用抬升量重复叠加到具体机型。
# 输入：
#   semantic：当前地图语义及碰撞基元。
#   vehicle：当前机型的碰撞中心偏移和机身高度。
# 输出：
#   root：机体直立静置于支撑面时的模型根 ENU 坐标。
def resolve_grounded_vehicle_model_root(
    semantic: dict[str, Any], vehicle: VehicleAsset
) -> Vector3:
    """Resolve a vehicle-specific resting pose from map and body geometry.

    A map cannot prescribe one model-root Z for every aircraft: model roots,
    landing gear and collision centres differ.  The horizontal spawn binding
    identifies the intended support location.  The selected aircraft then
    supplies the collision envelope needed to place its lower face on the
    highest terrain surface under that location.
    """

    bindings = load_map_runtime_bindings(semantic)
    offset = resolve_vehicle_collision_center_offset(vehicle)
    if not isinstance(vehicle, VehicleAsset) or not math.isfinite(vehicle.body_height_m):
        raise MapRuntimeBindingsError("VEHICLE_GROUNDED_POSE_INVALID")
    primitives = semantic.get("runtime_collision_primitives", semantic.get("collision_primitives"))
    if not isinstance(primitives, list) or not 1 <= len(primitives) <= 100_000:
        raise MapRuntimeBindingsError("MAP_SUPPORT_GEOMETRY_MISSING")
    x = bindings.vehicle_spawn.x
    y = bindings.vehicle_spawn.y
    launch_z = bindings.mission_launch_waypoint.z
    support_tops: list[float] = []
    for primitive in primitives:
        if not isinstance(primitive, dict) or primitive.get("semantic") != "terrain":
            continue
        values = tuple(
            primitive.get(key)
            for key in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
        )
        yaw = primitive.get("yaw_rad", 0.0)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise MapRuntimeBindingsError("MAP_SUPPORT_GEOMETRY_INVALID")
        if type(yaw) not in (int, float) or not math.isfinite(yaw):
            raise MapRuntimeBindingsError("MAP_SUPPORT_GEOMETRY_INVALID")
        center_x, center_y, center_z, size_x, size_y, size_z = map(float, values)
        if min(size_x, size_y, size_z) <= 0.0:
            raise MapRuntimeBindingsError("MAP_SUPPORT_GEOMETRY_INVALID")
        dx, dy = x - center_x, y - center_y
        cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        if abs(local_x) <= size_x / 2.0 + 1e-9 and abs(local_y) <= size_y / 2.0 + 1e-9:
            top = center_z + size_z / 2.0
            if top <= launch_z + 1e-9:
                support_tops.append(top)
    if not support_tops:
        raise MapRuntimeBindingsError("MAP_SUPPORT_SURFACE_NOT_FOUND_AT_VEHICLE_SPAWN")
    support_z = max(support_tops)
    collision_center_z = support_z + vehicle.body_height_m / 2.0
    root = Vector3(
        x=x - offset.x,
        y=y - offset.y,
        z=collision_center_z - offset.z,
    )
    if not all(math.isfinite(value) for value in (root.x, root.y, root.z)):
        raise MapRuntimeBindingsError("VEHICLE_GROUNDED_POSE_INVALID")
    return root
