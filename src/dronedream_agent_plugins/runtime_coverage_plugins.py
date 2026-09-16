"""Bounded convex-region coverage proposals, not collision-checked flight commands.

Both endpoints and every connecting segment remain in the inset convex region.
Concave/self-intersecting regions require decomposition by a different planner;
silently dropping outside vertices is not a safe substitute for that operation.
"""

from __future__ import annotations

import math
from typing import Any

from dronedream_agent_core.contracts import CoveragePattern, CoveragePlanRequest, Vector3
from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin

Point = tuple[float, float]
_EPSILON_M = 1e-8
_MAX_POINTS = 10_000


# 功能：
#   计算已验证覆盖区域的轴对齐外包范围，不将外包矩形当作可飞净空证明。
# 输入：
#   request：包含凸多边形或矩形中心与尺寸的覆盖请求。
# 输出：
#   bounds：依次为最小 x、最大 x、最小 y、最大 y 的米制范围。
def _bounds(request: CoveragePlanRequest) -> tuple[float, float, float, float]:
    if request.polygon_enu_m:
        xs = [point.x for point in request.polygon_enu_m]
        ys = [point.y for point in request.polygon_enu_m]
        bounds = (min(xs), max(xs), min(ys), max(ys))
        return bounds
    half_width = request.width_m / 2.0
    half_height = request.height_m / 2.0
    bounds = (
        request.center_enu_m.x - half_width,
        request.center_enu_m.x + half_width,
        request.center_enu_m.y - half_height,
        request.center_enu_m.y + half_height,
    )
    return bounds


# 功能：
#   计算有向平面面积，先平移到首顶点再精确求和，减轻大坐标相减造成的精度损失。
# 输入：
#   polygon：顺序排列的二维顶点。
# 输出：
#   area：有符号平方米面积，少于三个点时为零。
def _area(polygon: list[Point]) -> float:
    area = 0.0
    if len(polygon) < 3:
        return area
    ox, oy = polygon[0]
    area = (
        math.fsum(
            (a[0] - ox) * (b[1] - oy) - (b[0] - ox) * (a[1] - oy)
            for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True)
        )
        / 2
    )
    return area


# 功能：
#   求点到逆时针边内侧半平面的有符号距离；上游必须已拒绝零长边。
# 输入：
#   point：待检查二维点。
#   start：当前边起点。
#   end：当前边终点。
# 输出：
#   distance：内侧为正、外侧为负的垂距，单位米。
def _edge_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = (dx * (point[1] - start[1]) - dy * (point[0] - start[0])) / math.hypot(dx, dy)
    return distance


# 功能：
#   1. 在几何运算前重新验证可变请求的有限性、数量和范围。
#   2. 统一逆时针凸多边形；退化、自交或凹区域不得静默转成不安全连接。
# 输入：
#   request：待检查的覆盖请求。
# 输出：
#   region：重新构造的请求与逆时针凸区域顶点组成的元组。
def _region(request: CoveragePlanRequest) -> tuple[CoveragePlanRequest, list[Point]]:
    if len(request.polygon_enu_m) > 128:
        raise ValueError("COVERAGE_POLYGON_VERTEX_LIMIT")
    values = [
        request.width_m,
        request.height_m,
        request.lane_spacing_m,
        request.boundary_margin_m,
        request.altitude_m,
    ]
    values.extend(
        value
        for point in [request.center_enu_m, *request.polygon_enu_m]
        for value in (point.x, point.y, point.z)
    )
    if any(type(value) not in (int, float) or not -1e7 <= value <= 1e7 for value in values):
        raise ValueError("COVERAGE_NONFINITE_OR_UNBOUNDED_INPUT")
    request = CoveragePlanRequest.model_validate(request.model_dump(mode="python"))
    min_x, max_x, min_y, max_y = _bounds(request)
    if max_x - min_x > 5000 or max_y - min_y > 5000:
        raise ValueError("COVERAGE_REGION_EXTENT_LIMIT")
    polygon = [(point.x, point.y) for point in request.polygon_enu_m] or [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]
    # 允许常见的首尾闭合重复点，其他重复点或零长边不能被当作有效覆盖区域。
    if len(polygon) > 3 and polygon[0] == polygon[-1]:
        polygon.pop()
    if len(set(polygon)) != len(polygon):
        raise ValueError("COVERAGE_REGION_DEGENERATE")
    area = _area(polygon)
    if abs(area) <= _EPSILON_M:
        raise ValueError("COVERAGE_REGION_DEGENERATE")
    if area < 0:
        polygon.reverse()
    for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if math.dist(start, end) <= _EPSILON_M:
            raise ValueError("COVERAGE_REGION_DEGENERATE")
        if any(_edge_distance(point, start, end) < -_EPSILON_M for point in polygon):
            raise ValueError("COVERAGE_NONCONVEX_REGION_REQUIRES_DECOMPOSITION")
    region = (request, polygon)
    return region


# 功能：
#   求各边内偏半平面的交集，按实际倾斜边界留出距离，而不是只缩小外包矩形。
# 输入：
#   polygon：已验证的逆时针凸多边形。
#   margin：需要内缩的米制边距。
# 输出：
#   result：内缩后的凸区域顶点，区域消失或退化时为空列表。
def _inset(polygon: list[Point], margin: float) -> list[Point]:
    result = list(polygon)
    for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        clipped: list[Point] = []
        for a, b in zip(result, result[1:] + result[:1], strict=True):
            da = _edge_distance(a, start, end) - margin
            db = _edge_distance(b, start, end) - margin
            if da >= 0:
                clipped.append(a)
            if (da >= 0) != (db >= 0):
                t = da / (da - db)
                clipped.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        result = []
        for point in clipped:
            if not result or math.dist(result[-1], point) > _EPSILON_M:
                result.append(point)
        if len(result) > 1 and math.dist(result[0], result[-1]) <= _EPSILON_M:
            result.pop()
        if len(result) < 3:
            result.clear()
            return result
    if _area(result) <= _EPSILON_M:
        result.clear()
    return result


# 功能：
#   检查点数及凸区域连续包含关系后封装覆盖提案，不替代三维障碍或动态飞行验证。
# 输入：
#   request：已验证覆盖请求。
#   polygon：原始逆时针凸区域。
#   inset：按安全边距内缩的区域。
#   points：待连接的二维提案点。
#   name：覆盖模式名称。
#   lanes：生成的横向通道或内缩环数量。
# 输出：
#   result：通过覆盖契约校验的 JSON 提案。
def _pattern(
    request: CoveragePlanRequest,
    polygon: list[Point],
    inset: list[Point],
    points: list[Point],
    *,
    name: str,
    lanes: int,
) -> dict[str, object]:
    if not 2 <= len(points) <= _MAX_POINTS:
        raise ValueError("COVERAGE_PATTERN_POINT_LIMIT")
    # 凸集合包含任意两内点间的完整线段，逐一检查原始偏移边也就约束了连接段。
    if any(
        _edge_distance(point, start, end) < request.boundary_margin_m - _EPSILON_M
        for point in points
        for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True)
    ):
        raise ValueError("COVERAGE_SEGMENT_CONTAINMENT_FAILED")
    pattern = CoveragePattern(
        pattern=name,
        points_enu_m=[Vector3(x=x, y=y, z=request.altitude_m) for x, y in points],
        lane_count=lanes,
        estimated_area_m2=_area(inset),
        deterministic_gates={
            "bounded_region": True,
            "positive_spacing": True,
            "finite_points": True,
            "continuous_region_containment": True,
            "alternating_lanes" if name == "lawnmower" else "inward_progression": True,
        },
    )
    result = pattern.model_dump(mode="json")
    return result


# 功能：
#   在内缩凸区域内裁剪交替方向的水平通道，以数量预算限制生成成本。
# 输入：
#   request：区域、边距、高度及通道间距请求。
#   _：往复提案器不使用的扩展参数。
# 输出：
#   result：按边距与点数检查后的往复覆盖提案。
def _lawnmower(*, request: CoveragePlanRequest, **_: Any) -> dict[str, object]:
    request, polygon = _region(request)
    inset = _inset(polygon, request.boundary_margin_m)
    if not inset:
        raise ValueError("COVERAGE_MARGIN_COLLAPSES_REGION")
    min_y, max_y = min(y for _, y in inset), max(y for _, y in inset)
    lane_count = max(2, int(math.ceil((max_y - min_y) / request.lane_spacing_m)) + 1)
    if lane_count * 2 > _MAX_POINTS:
        raise ValueError("COVERAGE_PATTERN_POINT_LIMIT")
    points: list[Point] = []
    for lane in range(lane_count):
        y = min_y + lane * (max_y - min_y) / (lane_count - 1)
        if lane == lane_count - 1:
            y = max_y
        intersections: list[float] = []
        for a, b in zip(inset, inset[1:] + inset[:1], strict=True):
            if min(a[1], b[1]) <= y <= max(a[1], b[1]):
                if a[1] == b[1]:
                    intersections.extend((a[0], b[0]))
                else:
                    intersections.append(a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]))
        if not intersections:
            raise ValueError("COVERAGE_LANE_INTERSECTION_FAILED")
        ends = [min(intersections), max(intersections)]
        if lane % 2:
            ends.reverse()
        for x in ends:
            if not points or math.dist(points[-1], (x, y)) > _EPSILON_M:
                points.append((x, y))
    result = _pattern(
        request,
        polygon,
        inset,
        points,
        name="lawnmower",
        lanes=lane_count,
    )
    return result


# 功能：
#   生成逐层内缩的多边形环并连接，超过预算时明确拒绝，不能静默截断覆盖范围。
# 输入：
#   request：区域、边距、高度及层间距请求。
#   _：内缩提案器不使用的扩展参数。
# 输出：
#   result：有界且连续位于内缩区域中的环式覆盖提案。
def _spiral(*, request: CoveragePlanRequest, **_: Any) -> dict[str, object]:
    request, polygon = _region(request)
    inset = _inset(polygon, request.boundary_margin_m)
    if not inset:
        raise ValueError("COVERAGE_MARGIN_COLLAPSES_REGION")
    min_x, max_x, min_y, max_y = _bounds(request)
    maximum_rings = math.ceil(min(max_x - min_x, max_y - min_y) / (2 * request.lane_spacing_m))
    if maximum_rings > _MAX_POINTS // 3:
        raise ValueError("COVERAGE_PATTERN_POINT_LIMIT")
    points: list[Point] = []
    rings = 0
    # 有界整数索引避免反复累加浮点间距停滞，也不允许达到上限后假装覆盖已完成。
    for index in range(maximum_rings + 1):
        ring = _inset(polygon, request.boundary_margin_m + index * request.lane_spacing_m)
        if not ring:
            break
        if len(points) + len(ring) + 1 > _MAX_POINTS:
            raise ValueError("COVERAGE_PATTERN_POINT_LIMIT")
        points.extend([*ring, ring[0]])
        rings += 1
    result = _pattern(
        request,
        polygon,
        inset,
        points,
        name="inward-spiral",
        lanes=rings,
    )
    return result


# 功能：
#   注册互斥覆盖提案器，并要求在稳定悬停后才能切换；提案本身不授权采纳。
# 输入：
#   plugin_id：提案器插件标识。
#   name：展示名称。
#   description：支持区域及边界说明。
#   handler：覆盖提案函数。
#   enabled：默认启用状态。
#   order：插件展示顺序。
# 输出：
#   definition：覆盖提案插件的清单及执行钩子。
def _definition(
    *, plugin_id: str, name: str, description: str, handler: Any, enabled: bool, order: int
) -> PluginDefinition:
    definition = hook_plugin(
        module_name=__name__,
        plugin_id=plugin_id,
        name=name,
        description=description,
        capability_id=f"{plugin_id}.plan",
        capability_kind="runtime-replanner",
        capability_name=name,
        capability_description=description,
        category_id="runtime",
        category_label="运行期与在线换路",
        slot_id="runtime.coverage-planner",
        slot_label="区域覆盖轨迹",
        activation_mode="single",
        category_order=70,
        slot_order=35,
        plugin_order=order,
        hooks={"plan_coverage": handler},
        default_enabled=enabled,
        failure_mode="fail-closed",
        swap_policy="safe-hold",
        configuration_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    return definition


# 功能：
#   提供往复式与内缩环式两种凸区域提案，明确凹区域需要先由其他模块分解。
# 输入：
#   无。
# 输出：
#   definitions：稳定标识对应的覆盖插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions = [
        _definition(
            plugin_id="runtime.coverage-lawnmower",
            name="往复覆盖",
            description="以平行往复航线覆盖矩形或凸多边形；凹多边形须先分解。",
            handler=_lawnmower,
            enabled=True,
            order=10,
        ),
        _definition(
            plugin_id="runtime.coverage-inward-spiral",
            name="内收螺旋覆盖",
            description="在凸多边形内按真实边距逐层内收；仍须后续障碍物与飞行验收。",
            handler=_spiral,
            enabled=False,
            order=20,
        ),
    ]
    return definitions
