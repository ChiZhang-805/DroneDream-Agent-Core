"""Exact geometry counterexamples and content-binding tests, not flight evidence."""

import hashlib
import json
import math

import numpy as np
import pytest

from dronedream_agent_core import collision
from dronedream_agent_core.collision_batch import static_point_clearances
from dronedream_agent_core.contracts import GraphRoute, Vector3
from dronedream_agent_core.training.swept_geometry import SweptMapGeometry


# 功能：
#   构造一条零至一米的两点路线，用于暴露中间空间漏检。
# 输入：
#   无：固定合成路线。
# 输出：
#   route：没有飞行资格标记的测试路线。
def route():
    route = GraphRoute(
        start_node="a",
        goal_node="b",
        node_ids=["a", "b"],
        edge_ids=["a-b"],
        positions_m=[Vector3(x=0, y=0, z=0), Vector3(x=1, y=0, z=0)],
        route_length_m=1,
        all_edges_flight_verified=False,
    )
    return route


# 功能：
#   写入采样中点的一面薄墙，墙厚远小于测试采样间隔。
# 输入：
#   path：测试语义文件路径。
# 输出：
#   path：已写入薄墙几何的路径。
def semantic(path):
    path.write_text(
        json.dumps(
            {
                "collision_primitives": [
                    dict(
                        name="thin-wall",
                        center_x=0.5,
                        center_y=0,
                        center_z=0,
                        size_x=0.0001,
                        size_y=2,
                        size_z=2,
                    )
                ]
            }
        )
    )
    return path


# 功能：
#   两端都安全不能代表线段安全，粗采样也必须保守拒绝从薄墙中间穿过的路线。
# 输入：
#   tmp_path：测试语义目录。
# 输出：
#   None：不返回业务数据。
def test_thin_wall_between_samples_cannot_pass_route_clearance(tmp_path):
    report = collision.validate_route_clearance(
        route(),
        semantic(tmp_path / "map.json"),
        vehicle_diameter_m=0.01,
        vehicle_height_m=0.01,
        sample_interval_m=1,
    )
    assert not report.accepted
    assert report.minimum_clearance_m < 0
    assert report.minimum_clearance_m == min(report.segment_minimum_clearances_m)


# 功能：
#   几何文件在计算期间被替换后，回执仍绑定实际消费的原字节，不能指向未参与计算的新地图。
# 输入：
#   tmp_path：测试地图目录。
#   monkeypatch：在第一次净空计算后替换地图内容的工具。
# 输出：
#   None：不返回业务数据。
def test_route_receipt_hashes_the_actual_geometry_snapshot(tmp_path, monkeypatch):
    path = semantic(tmp_path / "map.json")
    original = path.read_bytes()
    real = collision._clearance

    # 功能：
    #   计算实际净空后改变原路径，以确认最终摘要不会重新读取另一版地图。
    # 输入：
    #   args：真实几何位置参数。
    #   kwargs：真实包络关键字参数。
    # 输出：
    #   value：真实净空函数返回的值。
    def replace_after_query(*args, **kwargs):
        value = real(*args, **kwargs)
        path.write_text('{"collision_primitives": []}')
        return value

    monkeypatch.setattr(collision, "_clearance", replace_after_query)
    report = collision.validate_route_clearance(
        route(), path, vehicle_diameter_m=0.01, vehicle_height_m=0.01, sample_interval_m=1
    )
    assert report.semantic_sha256 == hashlib.sha256(original).hexdigest()


# 功能：
#   路线容差和尺度不能通过布尔或无穷值绕过碰撞检查。
# 输入：
#   tmp_path：测试地图目录。
#   option：非法的尺度或容差。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "option",
    [
        {"vehicle_diameter_m": True},
        {"sample_interval_m": True},
        {"penetration_tolerance_m": math.inf},
        {"penetration_tolerance_m": -0.1},
    ],
)
def test_route_clearance_rejects_invalid_physical_options(tmp_path, option):
    arguments = dict(vehicle_diameter_m=0.1, vehicle_height_m=0.1)
    arguments.update(option)
    with pytest.raises(ValueError):
        collision.validate_route_clearance(route(), semantic(tmp_path / "map.json"), **arguments)


# 功能：
#   已旋转至水平的高箱体必须覆盖其横向空间，三个几何入口都不能把它当成原竖直箱体。
# 输入：
#   implementation：标量、批量或训练独立扫掠查询入口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("implementation", ["scalar", "batch", "swept"])
def test_tilted_box_is_not_mistaken_for_upright_geometry(implementation):
    primitive = dict(
        center_x=0, center_y=0, center_z=0, size_x=0.1, size_y=0.1, size_z=2, pitch_rad=math.pi / 2
    )
    point = (0.8, 0, 0)
    if implementation == "scalar":
        distance = collision.vehicle_clearance(point, primitive, radius_m=0.01, half_height_m=0.01)
    elif implementation == "batch":
        distance = static_point_clearances([point], [primitive], radius_m=0.01, half_height_m=0.01)[
            0
        ][0]
    else:
        distance = SweptMapGeometry([primitive], radius_m=0.01, half_height_m=0.01).clearance(
            [point, point]
        )
    assert distance < 0


# 功能：
#   点净空查询拒绝错误尺寸和坐标，不把非有限值或真假开关解释为几何。
# 输入：
#   value：非法物理量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_scalar_geometry_rejects_invalid_dimensions(value):
    with pytest.raises(ValueError):
        collision.vehicle_clearance(
            (0, 0, 0),
            dict(center_x=2, center_y=0, center_z=0, radius_m=0.1),
            radius_m=value,
            half_height_m=0.1,
        )


# 功能：
#   用独立旋转矩阵生成真实箱体八个角点，验证任意复合姿态的保守包络没有漏掉角点。
# 输入：
#   angles：滚转、俯仰和偏航三个弧度角。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("angles", [(0.2, -0.4, 1.1), (1.2, 0.8, -2.0), (0.0, 0.0, 0.0)])
def test_box_envelope_covers_independently_rotated_corners(angles):
    roll, pitch, yaw = angles
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rotation = (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )
    shape = dict(
        center_x=0,
        center_y=0,
        center_z=0,
        size_x=0.2,
        size_y=1.4,
        size_z=3,
        roll_rad=roll,
        pitch_rad=pitch,
        yaw_rad=yaw,
    )
    points = [
        tuple(float(value) for value in rotation @ np.array([x, y, z]))
        for x in (-0.1, 0.1)
        for y in (-0.7, 0.7)
        for z in (-1.5, 1.5)
    ]
    scalar = [
        collision.vehicle_clearance(point, shape, radius_m=0.01, half_height_m=0.01)
        for point in points
    ]
    batched, indices = static_point_clearances(points, [shape], radius_m=0.01, half_height_m=0.01)
    assert max(scalar) < 0
    assert set(indices) == {0}
    np.testing.assert_allclose(batched, scalar, atol=1e-12, rtol=0)


# 功能：
#   极小采样间隔在生成列表之前拒绝，不靠分配大量内存后才发现预算耗尽。
# 输入：
#   tmp_path：测试地图目录。
# 输出：
#   None：不返回业务数据。
def test_unbounded_sampling_is_rejected_before_allocation(tmp_path):
    with pytest.raises(ValueError, match="sample budget"):
        collision.validate_route_clearance(
            route(),
            semantic(tmp_path / "map.json"),
            vehicle_diameter_m=0.1,
            vehicle_height_m=0.1,
            sample_interval_m=1e-320,
        )


# 功能：
#   重复地图键或混合两种形状的歧义基元不能参与路线验收。
# 输入：
#   tmp_path：测试地图目录。
#   ambiguous：是否使用混合形状；否则构造重复顶层键。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("ambiguous", [False, True])
def test_ambiguous_map_content_is_rejected(tmp_path, ambiguous):
    path = semantic(tmp_path / "map.json")
    if ambiguous:
        value = json.loads(path.read_text())
        value["collision_primitives"][0]["radius_m"] = 1.0
        path.write_text(json.dumps(value))
    else:
        path.write_text('{"collision_primitives": [],' + path.read_text()[1:])
    with pytest.raises(ValueError):
        collision.validate_route_clearance(
            route(), path, vehicle_diameter_m=0.1, vehicle_height_m=0.1
        )


# 功能：
#   批量位置不能通过转浮点数掩盖字符串、布尔或复数坐标。
# 输入：
#   points：具有非法原始类型的测试坐标。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "points",
    [
        [[True, 0, 0]],
        [["1", "0", "0"]],
        np.array([[1 + 2j, 0, 0]]),
        np.array([[True, False, False]]),
    ],
)
def test_batch_coordinates_are_not_silently_coerced(points):
    with pytest.raises(ValueError, match="INPUT_INVALID"):
        static_point_clearances(points, [], radius_m=0.1, half_height_m=0.1)
