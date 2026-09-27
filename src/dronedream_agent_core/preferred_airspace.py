"""Map-wide soft 3D flight preferences; no route, sensor or flight-qualification authority."""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from .collision import primitive_bounds
from .hashing import sha256_json

Point = tuple[float, float, float]
Bounds = tuple[Point, Point]
VERSION = "dronedream.preferred-airspace.v1"


@dataclass(frozen=True)
class AirspacePreferences:
    resolution_m: float = 0.5
    margin_m: float = 0.3
    indoor_lower_fraction: float = 0.4
    indoor_upper_fraction: float = 0.8
    outdoor_lower_agl_m: float = 5.0
    outdoor_upper_agl_m: float = 8.0

    # 功能：
    #   验证有限、可排序的软偏好；分辨率限制同时约束全图生成与运行时缓存规模。
    # 输入：
    #   self：空间偏好。
    # 输出：
    #   None：无业务返回值。
    def __post_init__(self) -> None:
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("AIRSPACE_PREFERENCES_NONFINITE")
        if not (0.25 <= self.resolution_m <= 2 and 0 <= self.margin_m <= 3
                and 0 <= self.indoor_lower_fraction < self.indoor_upper_fraction <= 1
                and 0 < self.outdoor_lower_agl_m < self.outdoor_upper_agl_m <= 50):
            raise ValueError("AIRSPACE_PREFERENCES_INVALID")


@dataclass(frozen=True)
class HeightBand:
    floor_z_m: float
    ceiling_z_m: float | None
    minimum_center_z_m: float
    maximum_center_z_m: float
    preferred_lower_z_m: float
    preferred_upper_z_m: float


# 功能：
#   验证查询坐标，避免 NaN、布尔和极端数值进入索引或模型特征。
# 输入：
#   point：世界 ENU 米制坐标。
# 输出：
#   checked：有限的三维坐标。
def checked_point(point) -> Point:
    if (not isinstance(point, (tuple, list)) or len(point) != 3
            or any(type(v) not in (int, float) or abs(v) > 1e6 or not math.isfinite(v)
                   for v in point)):
        raise ValueError("AIRSPACE_POINT_INVALID")
    checked = tuple(float(v) for v in point)
    return checked


class PreferredAirspace:
    """Sparse XY columns with multiple free height intervals, independent of any mission."""

    # 功能：
    #   1. 从绑定地图的真实碰撞图元建立全图列索引；空白或无覆盖声明的地图不自动变为自由空间。
    #   2. 用完整机体和余量扩张障碍，保留每层上下边界；缓存与机型、负载和偏好绑定。
    # 输入：
    #   semantic：地图语义快照。
    #   binding：上游核验的地图／机型／负载标识。
    #   radius_m、height_m：当前机体与已知负载的保守碰撞包络。
    #   preferences：经过验证的软偏好。
    # 输出：
    #   self：只提供静态偏好的空间场，不授予运动权限。
    def __init__(self, semantic: dict, binding: dict, radius_m: float, height_m: float,
                 preferences: AirspacePreferences | None = None):
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 6
               for v in (radius_m, height_m)):
            raise ValueError("AIRSPACE_ENVELOPE_INVALID")
        self.preferences = preferences or AirspacePreferences()
        self.radius = radius_m
        self.half_height = height_m / 2
        self.binding = {"sources": binding, "semantic_sha256": sha256_json(semantic),
                        "radius_m": radius_m, "height_m": height_m,
                        "preferences": asdict(self.preferences), "schema_version": VERSION}
        self.sha256 = sha256_json(self.binding)
        primitives = semantic.get("collision_primitives", [])
        runtime = semantic.get("runtime_collision_primitives", [])
        if (not isinstance(primitives, list) or not isinstance(runtime, list)
                or not 1 <= len(primitives) + len(runtime) <= 100_000):
            raise ValueError("AIRSPACE_GEOMETRY_INVALID")
        # 两层取并集：渲染精细几何和实际仿真保守包络均不能被另一层漏掉。
        self.bounds = [primitive_bounds(item) for item in (*primitives, *runtime)]
        self._index: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._columns: OrderedDict[tuple[int, int], tuple[HeightBand, ...]] = OrderedDict()
        terrain = [bound for item, bound in zip((*primitives, *runtime), self.bounds, strict=True)
                   if item.get("semantic") == "terrain"]
        self.domain = None
        if terrain and semantic.get("coordinate_frame") == "ENU":
            self.domain = (tuple(min(b[0][i] for b in terrain) for i in range(2)),
                           tuple(max(b[1][i] for b in terrain) for i in range(2)))
        self._terrain = terrain
        if self.domain is None:
            return
        step = self.preferences.resolution_m
        self.first = tuple(math.floor(v / step) for v in self.domain[0])
        self.last = tuple(math.ceil(v / step) - 1 for v in self.domain[1])
        if math.prod(self.last[i] - self.first[i] + 1 for i in range(2)) > 200_000:
            raise ValueError("AIRSPACE_COLUMN_BUDGET_EXCEEDED")
        reach = self.radius + self.preferences.margin_m
        references = 0
        for index, (low, high) in enumerate(self.bounds):
            first = [max(self.first[i], math.floor((low[i] - reach) / step)) for i in range(2)]
            last = [min(self.last[i], math.floor((high[i] + reach) / step)) for i in range(2)]
            references += math.prod(max(0, last[i] - first[i] + 1) for i in range(2))
            if references > 3_000_000:
                raise ValueError("AIRSPACE_INDEX_BUDGET_EXCEEDED")
            for x in range(first[0], last[0] + 1):
                for y in range(first[1], last[1] + 1):
                    self._index[x, y].append(index)

    # 功能：
    #   逐列扣除膨胀障碍并保留多楼层区间；地形覆盖外、无地面或贴边格子保持未知。
    # 输入：
    #   key：全图 XY 网格键。
    # 输出：
    #   bands：该格子全面积可采用的静态高度偏好。
    def column(self, key: tuple[int, int]) -> tuple[HeightBand, ...]:
        if self.domain is None:
            return ()
        if key in self._columns:
            self._columns.move_to_end(key)
            return self._columns[key]
        step, margin = self.preferences.resolution_m, self.preferences.margin_m
        x, y = key[0] * step, key[1] * step
        reach = self.radius + margin
        covered_terrain = [high[2] for low, high in self._terrain
                          if low[0] <= x - reach and high[0] >= x + step + reach
                   and low[1] <= y - reach and high[1] >= y + step + reach
                          ]
        if not covered_terrain:
            return ()
        occupied = []
        for index in self._index.get(key, ()):
            low, high = self.bounds[index]
            if (low[0] <= x + step + reach and high[0] >= x - reach
                    and low[1] <= y + step + reach and high[1] >= y - reach):
                occupied.append((low[2], high[2]))
        occupied.sort()
        merged: list[list[float]] = []
        for low, high in occupied:
            if merged and low <= merged[-1][1]:
                merged[-1][1] = max(high, merged[-1][1])
            else:
                merged.append([low, high])
        bands = []
        for index, (_, floor) in enumerate(merged):
            ceiling = merged[index + 1][0] if index + 1 < len(merged) else None
            lower = floor + self.half_height + margin
            upper = (ceiling - self.half_height - margin if ceiling is not None
                     else floor + self.preferences.outdoor_upper_agl_m)
            if upper <= lower:
                continue
            if ceiling is None:
                # AGL 参考真实地形，不把树冠或灯柱顶端错误地当成新地面继续抬升 5–8 米。
                ground = max(covered_terrain)
                preferred_low = max(lower, ground + self.preferences.outdoor_lower_agl_m)
                preferred_high = ground + self.preferences.outdoor_upper_agl_m
            else:
                preferred_low = lower + (upper - lower) * self.preferences.indoor_lower_fraction
                preferred_high = lower + (upper - lower) * self.preferences.indoor_upper_fraction
            if preferred_high > preferred_low:
                bands.append(HeightBand(
                    floor, ceiling, lower, upper, preferred_low, preferred_high))
        result = tuple(bands)
        self._columns[key] = result
        if len(self._columns) > 200_000:
            self._columns.popitem(last=False)
        return result

    # 功能：
    #   查询当前位置所属静态高度层；楼板内不跳到另一层，空间未知不伪造高度。
    # 输入：
    #   point：实际机体碰撞中心。
    # 输出：
    #   band：所在高度层或 None。
    def band_at(self, point: Point) -> HeightBand | None:
        point = checked_point(point)
        key = tuple(math.floor(v / self.preferences.resolution_m) for v in point[:2])
        band = next((band for band in self.column(key)
                     if band.minimum_center_z_m <= point[2]
                     and (band.ceiling_z_m is None or point[2] <= band.maximum_center_z_m)), None)
        return band

    # 功能：
    #   输出归一化的软偏离代价；不能以零代价解释为未知空间可以飞行。
    # 输入：
    #   point：规划采样中心。
    # 输出：
    #   cost：非负且有界的偏离代价。
    def cost(self, point: Point) -> float:
        band = self.band_at(point)
        if band is None:
            return 1.0
        distance = max(band.preferred_lower_z_m - point[2],
                       point[2] - band.preferred_upper_z_m, 0.0)
        cost = min(4.0, distance / max(0.25, band.preferred_upper_z_m - band.preferred_lower_z_m))
        return cost

    # 功能：
    #   以相邻列的同层地面拟合楼梯总体斜率，只改变建议值，不删改真实阶梯碰撞。
    # 输入：
    #   point：当前碰撞中心。
    # 输出：
    #   context：绑定空间摘要、原始边界和建议坡度的结构化模型输入。
    def context(self, point: Point) -> dict[str, Any]:
        point = checked_point(point)
        band = self.band_at(point)
        result = {"schema_version": VERSION, "airspace_sha256": self.sha256,
                  "authority": "preference-only", "available": band is not None,
                  "coordinate_frame": "ENU", "query_position_m": list(point)}
        if band is None:
            return result
        step = self.preferences.resolution_m
        gradients = []
        for axis in range(2):
            samples = []
            for direction in (-1, 1):
                neighbor = list(point)
                neighbor[axis] += direction * step
                other = self.band_at(tuple(neighbor))
                if other is not None and abs(other.floor_z_m - band.floor_z_m) <= step:
                    samples.append((other.floor_z_m - band.floor_z_m) / (direction * step))
            gradients.append(sum(samples) / len(samples) if samples else 0.0)
        result.update({"height_band": asdict(band), "floor_gradient_xy": gradients,
                       "deviation_cost": self.cost(point), "requires_live_clearance": True})
        return result

    # 功能：
    #   生成全地图的可视化体积，沿 X 合并同层相邻格子；结果不依赖任何任务路线。
    # 输入：
    #   self：当前不可变地图与机体绑定的空间场。
    # 输出：
    #   snapshot：真实空间体积、保守障碍及可复验摘要。
    def snapshot(self) -> dict:
        volumes = []
        step = self.preferences.resolution_m
        if self.domain is not None:
            for y in range(self.first[1], self.last[1] + 1):
                active: dict[HeightBand, int] = {}
                for x in range(self.first[0], self.last[0] + 2):
                    current = set(self.column((x, y))) if x <= self.last[0] else set()
                    for band in active.keys() - current:
                        start = active.pop(band)
                        volumes.append([(start + x) * step / 2, (y + 0.5) * step,
                                        (band.preferred_lower_z_m + band.preferred_upper_z_m) / 2,
                                        (x - start) * step, step,
                                        band.preferred_upper_z_m - band.preferred_lower_z_m,
                                        band.floor_z_m, band.ceiling_z_m])
                    for band in current - active.keys():
                        active[band] = x
                    if len(volumes) > 200_000:
                        raise ValueError("AIRSPACE_RENDER_BUDGET_EXCEEDED")
        volumes.sort()
        snapshot = {"schema_version": VERSION, "binding": self.binding,
                    "airspace_sha256": self.sha256, "coordinate_frame": "ENU",
                    "authority": "preference-only", "volumes": volumes,
                    "obstacles": [[*low, *high] for low, high in self.bounds],
                    "volume_layout": "cx,cy,cz,size_x,size_y,size_z,floor_z,ceiling_z_or_null",
                    "limitations": ["static-map-only", "conservative-AABB-envelope",
                                    "unknown-outside-terrain", "live-safety-required"]}
        snapshot["snapshot_sha256"] = sha256_json(snapshot)
        return snapshot
