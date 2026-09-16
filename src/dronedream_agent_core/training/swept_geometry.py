"""Batched independent outcome geometry, with the scalar collision semantics.

This is a computation optimization, not a reduced map or a collision shortcut.
Every primitive and every conservative segment sample remains in the query.
"""

from __future__ import annotations

import math

import numpy as np

from ..collision import _axis_endpoints, _box_half_sizes, _validated_primitive
from .evidence_snapshot import detach_evidence
from .geometry_inputs import finite_metric, metric_vector


class SweptMapGeometry:
    """Frozen map queries for independent scoring, never privileged actor features."""

    # 功能：
    #   校验并独立保存全部地图基元，按箱体、圆柱、球体和胶囊体编译米制数组。
    # 输入：
    #   self：待初始化的批量扫掠查询器。
    #   primitives：一至十万个地图几何基元。
    #   radius_m、half_height_m：无人机的正水平半径及半高，米。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, primitives: list[dict], *, radius_m: float, half_height_m: float):
        if (
            type(primitives) is not list
            or not 1 <= len(primitives) <= 100_000
            or not all(finite_metric(v) and v > 0 for v in (radius_m, half_height_m))
        ):
            raise ValueError("OUTCOME_GEOMETRY_INVALID")
        try:
            self.primitives = [
                _validated_primitive(primitive) for primitive in detach_evidence(primitives)
            ]
        except ValueError as error:
            raise ValueError("OUTCOME_GEOMETRY_INVALID") from error
        self.radius_m, self.half_height_m = radius_m, half_height_m
        boxes = [p for p in self.primitives if "size_x" in p]
        self.cylinders, self.spheres, self.capsules = [], [], []
        for p in self.primitives:
            if "size_x" in p:
                continue
            center = [float(p[f"center_{a}"]) for a in "xyz"]
            radius = float(p["radius_m"])
            if not all(math.isfinite(v) for v in (*center, radius)) or radius <= 0:
                raise ValueError("OUTCOME_GEOMETRY_INVALID")
            upright = (
                abs(float(p.get("roll_rad", 0.0))) <= 1e-12
                and abs(float(p.get("pitch_rad", 0.0))) <= 1e-12
            )
            if "length_m" in p or "height_m" in p:
                length = float(p.get("length_m", p.get("height_m")))
                if not math.isfinite(length) or length <= 0:
                    raise ValueError("OUTCOME_GEOMETRY_INVALID")
                if "length_m" not in p and upright:
                    self.cylinders.append([*center, radius, length / 2])
                else:
                    start, end = _axis_endpoints(p, length)
                    self.capsules.append([*start, *end, radius])
            else:
                self.spheres.append([*center, radius])
        self.cylinders = np.asarray(self.cylinders, dtype=np.float64).reshape(-1, 5)
        self.spheres = np.asarray(self.spheres, dtype=np.float64).reshape(-1, 4)
        self.capsules = np.asarray(self.capsules, dtype=np.float64).reshape(-1, 7)
        if not all(
            np.isfinite(values).all() for values in (self.cylinders, self.spheres, self.capsules)
        ):
            raise ValueError("OUTCOME_GEOMETRY_INVALID")
        self.centers = np.asarray(
            [[p[f"center_{a}"] for a in "xyz"] for p in boxes], dtype=np.float64
        ).reshape(-1, 3)
        # 与运行时相同的旋转外包络，避免训练评估把倾斜箱体误当竖直障碍。
        self.half_sizes = np.asarray([_box_half_sizes(p) for p in boxes], dtype=np.float64).reshape(
            -1, 3
        )
        angles = np.asarray([p.get("yaw_rad", 0.0) for p in boxes], dtype=np.float64)
        if (
            not all(math.isfinite(v) and v > 0 for v in (radius_m, half_height_m))
            or not np.isfinite(self.centers).all()
            or not np.isfinite(angles).all()
            or not np.isfinite(self.half_sizes).all()
            or (self.half_sizes <= 0).any()
        ):
            raise ValueError("OUTCOME_GEOMETRY_INVALID")
        self.cosines, self.sines = np.cos(angles), np.sin(angles)

    # 功能：
    #   验证有界三维轨迹并计算保守扫掠距离，任何溢出都显式失败，不忽略出错的障碍块。
    # 输入：
    #   self：已编译且保留全部障碍的地图查询器。
    #   positions：两至一万个有限米制位置。
    # 输出：
    #   minimum：含采样间距扣减的最小有限安全距离下界，米。
    def clearance(self, positions: list[tuple[float, float, float]]) -> float:
        if type(positions) not in (list, tuple) or not 2 <= len(positions) <= 10_000:
            raise ValueError("OUTCOME_GEOMETRY_REQUIRES_BOUNDED_SEGMENTS")
        try:
            positions = [metric_vector(point) for point in positions]
        except ValueError as error:
            raise ValueError("OUTCOME_POSE_JUMP_NOT_VERIFIABLE") from error
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                minimum = self._clearance(positions)
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("OUTCOME_GEOMETRY_NUMERICAL_OVERFLOW") from error
        return minimum

    # 功能：
    #   按至多两厘米间距采样并逐块检查全部障碍，以半间隔扣减保持采样间隙的保守性。
    # 输入：
    #   self：已编译的全量地图。
    #   positions：外层已验证的有限三维位置。
    # 输出：
    #   minimum：静态扫掠最小下界，米。
    def _clearance(self, positions):
        samples, penalties = [], []
        for start, end in zip(positions, positions[1:], strict=False):
            distance = math.dist(start, end)
            if not math.isfinite(distance) or distance > 10:
                raise ValueError("OUTCOME_POSE_JUMP_NOT_VERIFIABLE")
            count = max(1, math.ceil(distance / 0.02))
            if len(samples) + count + 1 > 250_000:
                raise ValueError("OUTCOME_GEOMETRY_SAMPLE_BUDGET_EXCEEDED")
            for i in range(count + 1):
                samples.append(
                    tuple(a + (b - a) * i / count for a, b in zip(start, end, strict=True))
                )
                penalties.append(distance / (2 * count))
        if not samples:
            raise ValueError("OUTCOME_GEOMETRY_REQUIRES_SEGMENT")
        minimum = math.inf
        # Bound temporary arrays even for long windows and dense maps.
        for offset in range(0, len(samples), 128):
            points = np.asarray(samples[offset : offset + 128], dtype=np.float64)
            penalty = np.asarray(penalties[offset : offset + 128], dtype=np.float64)
            for first in range(0, len(self.centers), 1024):
                slab = slice(first, first + 1024)
                delta = points[:, None, :] - self.centers[None, slab, :]
                cosine, sine = self.cosines[slab], self.sines[slab]
                local_x = cosine * delta[:, :, 0] + sine * delta[:, :, 1]
                local_y = -sine * delta[:, :, 0] + cosine * delta[:, :, 1]
                outside_x = np.maximum(np.abs(local_x) - self.half_sizes[slab, 0], 0.0)
                outside_y = np.maximum(np.abs(local_y) - self.half_sizes[slab, 1], 0.0)
                horizontal = np.hypot(outside_x, outside_y) - self.radius_m
                vertical = np.abs(delta[:, :, 2]) - self.half_sizes[slab, 2] - self.half_height_m
                clearance = np.where(
                    (horizontal > 0) & (vertical > 0),
                    np.hypot(horizontal, vertical),
                    np.maximum(horizontal, vertical),
                )
                minimum = min(minimum, float(np.min(clearance - penalty[:, None])))
            enclosing_radius = math.hypot(self.radius_m, self.half_height_m)
            # Tilted cylinders/capsules and spheres use a vehicle-enclosing
            # sphere. This is deliberately conservative, not a learned score.
            for kind, values in (
                ("cylinder", self.cylinders),
                ("sphere", self.spheres),
                ("capsule", self.capsules),
            ):
                for first in range(0, len(values), 1024):
                    block = values[first : first + 1024]
                    delta = points[:, None, :] - block[None, :, :3]
                    if kind == "cylinder":
                        horizontal = np.hypot(delta[:, :, 0], delta[:, :, 1])
                        horizontal -= block[:, 3] + self.radius_m
                        vertical = np.abs(delta[:, :, 2]) - block[:, 4] - self.half_height_m
                        clearance = np.where(
                            (horizontal > 0) & (vertical > 0),
                            np.hypot(horizontal, vertical),
                            np.maximum(horizontal, vertical),
                        )
                    elif kind == "capsule":
                        axis = block[:, 3:6] - block[:, :3]
                        squared = np.sum(axis * axis, axis=1)
                        # Match the scalar degenerate-segment treatment.
                        ratio = np.sum(delta * axis[None, :, :], axis=2) / np.maximum(
                            squared, 1e-18
                        )
                        ratio = np.where(squared <= 1e-18, 0.0, np.clip(ratio, 0.0, 1.0))
                        relative = delta - ratio[:, :, None] * axis[None, :, :]
                        clearance = np.linalg.norm(relative, axis=2) - block[:, 6]
                        clearance -= enclosing_radius
                    else:
                        clearance = np.linalg.norm(delta, axis=2) - block[:, 3] - enclosing_radius
                    minimum = min(minimum, float(np.min(clearance - penalty[:, None])))
        if not math.isfinite(minimum):
            raise ValueError("OUTCOME_GEOMETRY_INVALID")
        return minimum
