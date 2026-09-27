"""Versioned asset-pair spatial view shared by desktop and mission context."""

from __future__ import annotations

import json
from collections import OrderedDict
from threading import Lock

from pydantic import Field

from dronedream_agent_core.contracts import StrictModel, VehicleAsset
from dronedream_agent_core.preferred_airspace import PreferredAirspace

from .asset_interpretation import _read_bound_json
from .asset_runtime_resolver import resolve_versioned_map, resolve_versioned_vehicle
from .storage import AppStore


class AirspaceRequest(StrictModel):
    map_asset_id: str = Field(min_length=1, max_length=160)
    map_content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    vehicle_asset_id: str = Field(min_length=1, max_length=160)
    vehicle_content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class AirspaceService:
    # 功能：
    #   创建资产对空间缓存，最多保留两份全图；不使用进程全局跨库缓存。
    # 输入：
    #   store：当前 Core 资产库。
    # 输出：
    #   self：串行且容量有界的空间服务。
    def __init__(self, store: AppStore):
        self.store = store
        self._lock = Lock()
        self._cache: OrderedDict[str, PreferredAirspace] = OrderedDict()

    # 功能：
    #   重读并核验选定地图、飞机文件后生成同一空间绑定；不修改资产、证书或实测负载。
    # 输入：
    #   request：精确地图／机型版本。
    # 输出：
    #   field：基于空载原始包络的静态空间偏好，飞行负载仍由运行时独立重算。
    def field(self, request: AirspaceRequest) -> PreferredAirspace:
        map_record = self.store.get_asset_version(request.map_asset_id, request.map_content_sha256)
        vehicle_record = self.store.get_asset_version(
            request.vehicle_asset_id, request.vehicle_content_sha256)
        selected_map = resolve_versioned_map(map_record)
        selected_vehicle = resolve_versioned_vehicle(vehicle_record)
        semantic = json.loads(_read_bound_json(
            map_record, selected_map.root, selected_map.semantic))
        vehicle = VehicleAsset.model_validate_json(_read_bound_json(
            vehicle_record, selected_vehicle.root, selected_vehicle.vehicle_metadata))
        field = PreferredAirspace(semantic, request.model_dump(mode="json"),
                                  vehicle.body_radius_m, vehicle.body_height_m)
        with self._lock:
            cached = self._cache.get(field.sha256)
            if cached is not None:
                self._cache.move_to_end(field.sha256)
                return cached
            self._cache[field.sha256] = field
            if len(self._cache) > 2:
                self._cache.popitem(last=False)
        return field

    # 功能：
    #   对完整空间序列化互斥，防止同一缓存被并发生成修改；返回值只包含可视化和软偏好。
    # 输入：
    #   request：已验证的地图／机型版本。
    # 输出：
    #   snapshot：与计划和机型绑定的全图空间数据。
    def snapshot(self, request: AirspaceRequest) -> dict:
        field = self.field(request)
        with self._lock:
            snapshot = field.snapshot()
        return snapshot
