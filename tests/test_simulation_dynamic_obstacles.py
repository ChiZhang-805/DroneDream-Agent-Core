"""Physical-geometry and timing checks for simulation-only observation paths."""

import ast
import math
from pathlib import Path

import pytest
from test_training_outcome_channel import entity

from dronedream_agent_core import gazebo_adapter
from dronedream_agent_core.simulation_dynamic_obstacles import (
    dynamic_observation,
    dynamic_positions,
    load_dynamic_geometry,
    simulation_velocity,
    witness_age,
)


# 功能：
#   写入具有明确链接、模型根及碰撞体的合成世界，不启动仿真进程。
# 输入：
#   tmp_path：隔离测试目录。
#   shape：碰撞几何 XML 内容。
#   link_pose：可选链接位姿片段。
#   collision_pose：可选碰撞局部位姿片段。
#   extra：模型层额外节点，用于拒绝用例。
# 输出：
#   path：当前测试 SDF 路径。
def world(tmp_path, shape, *, link_pose="", collision_pose="", extra=""):
    path = tmp_path / "scene.sdf"
    path.write_text(
        '<sdf version="1.9"><world name="scene"><model name="dronedream_dynamic_cart">'
        + extra
        + '<link name="body">'
        + link_pose
        + '<collision name="shape">'
        + collision_pose
        + "<geometry>"
        + shape
        + "</geometry></collision></link></model></world></sdf>",
        encoding="utf-8",
    )
    return path


# 功能：
#   检查当前尺寸及模型根位置参与计算，包络包含盒、球和柱全部旋转，没有旧行人偏移。
# 输入：
#   tmp_path：独立世界目录。
#   shape：受支持的物理基本几何。
#   radius：独立计算的预期包围球半径。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "shape,radius",
    [
        ("<box><size>0.2 0.2 0.8</size></box>", math.sqrt(0.18)),
        ("<sphere><radius>0.7</radius></sphere>", 0.7),
        ("<cylinder><radius>0.3</radius><length>0.8</length></cylinder>", 0.5),
    ],
)
def test_declared_geometry_not_human_template(tmp_path, shape, radius):
    radii = load_dynamic_geometry(world(tmp_path, shape))
    row = dynamic_observation(
        "dronedream_dynamic_cart", (1.0, 2.0, 9.0), (0.0, 0.0, 0.0), radii, velocity_known=True
    )
    assert row.radius_m == pytest.approx(radius)
    assert row.height_m == pytest.approx(2 * radius)
    assert row.position_m.z == 9.0 and row.confidence == 1.0


# 功能：
#   验证内部平移和旋转不使包络缩小，任意模型朝向均被保守球覆盖。
# 输入：
#   tmp_path：隔离世界目录。
# 输出：
#   None：不返回业务数据。
def test_local_offsets_expand_rotation_independent_bound(tmp_path):
    radii = load_dynamic_geometry(
        world(
            tmp_path,
            "<box><size>2 2 2</size></box>",
            link_pose="<pose>0 0 1 0 1.57 0</pose>",
            collision_pose="<pose>0 2 0 0 0 0</pose>",
        )
    )
    assert radii["dronedream_dynamic_cart"] == pytest.approx(3 + math.sqrt(3))


# 功能：
#   验证不支持或歧义碰撞数据明确拒绝，不能隐式省略物体后继续判安全。
# 输入：
#   tmp_path：隔离测试目录。
#   shape：无效几何。
#   link_pose：无法可靠解析的链接位置。
#   extra：非刚性模型结构。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "shape,link_pose,extra",
    [
        ("<mesh><uri>unknown.obj</uri></mesh>", "", ""),
        ("<box><size>-1 1 1</size></box>", "", ""),
        ("<sphere><radius>nan</radius></sphere>", "", ""),
        ("<sphere><radius>1</radius><radius>2</radius></sphere>", "", ""),
        ("<sphere><radius>21</radius></sphere>", "", ""),
        ("<sphere><radius>1</radius></sphere>", '<pose relative_to="other">0 0 0 0 0 0</pose>', ""),
        ("<sphere><radius>1</radius></sphere>", "", '<joint name="moving"/>'),
        ("<sphere><radius>1</radius></sphere>", "", '<link name="unattached"/>'),
    ],
)
def test_unsupported_geometry_fails_closed(tmp_path, shape, link_pose, extra):
    with pytest.raises(ValueError, match="GEOMETRY"):
        load_dynamic_geometry(world(tmp_path, shape, link_pose=link_pose, extra=extra))


# 功能：
#   检查重复模型归一 ID、文档类型声明及多个世界均无法成为可信几何。
# 输入：
#   tmp_path：隔离 SDF 路径。
#   content：畸形或含歧义的完整世界文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "content",
    [
        '<!DOCTYPE sdf [<!ENTITY x "data">]><sdf><world name="x"/></sdf>',
        '<sdf><world name="a"/><world name="b"/></sdf>',
        '<sdf><world name="a"><model name="person_X"><link name="a"><collision name="a">'
        "<geometry><sphere><radius>1</radius></sphere></geometry></collision></link></model>"
        '<model name="person_x"/></world></sdf>',
    ],
)
def test_world_identity_and_xml_boundaries(tmp_path, content):
    path = tmp_path / "invalid.sdf"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_dynamic_geometry(path)


# 功能：
#   验证连续仿真时间计算速度，非法或重复时间和数值溢出被拒绝。
# 输入：
#   无；使用已知位移、时差及极端值。
# 输出：
#   None：不返回业务数据。
def test_velocity_and_age_use_valid_explicit_clocks():
    assert simulation_velocity((0.2, 0.0, 0.0), (0, (0.0, 0.0, 0.0)), 100_000_000) == (
        2.0,
        0.0,
        0.0,
    )
    assert witness_age(10.0, 10.125) == 0.125
    for stamp in (True, -1, 0, 2**63):
        with pytest.raises(ValueError):
            simulation_velocity((0.0, 0.0, 0.0), (0, (0.0, 0.0, 0.0)), stamp)
    with pytest.raises(ValueError):
        simulation_velocity((1e308, 0.0, 0.0), (0, (-1e308, 0.0, 0.0)), 1)
    for processed in (9.0, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            witness_age(10.0, processed)


# 功能：
#   验证动态坐标独立保存，布尔坐标、超量实体和未知尺寸均不可进入结果。
# 输入：
#   无；使用合成位姿列表。
# 输出：
#   None：不返回业务数据。
def test_dynamic_positions_are_bounded_owned_numeric_values():
    source = entity("person_worker", 1.0, 2.0, 3.0)
    positions = dynamic_positions([source])
    source.position.x = 100.0
    assert positions["person_worker"] == (1.0, 2.0, 3.0)
    source.position.x = True
    with pytest.raises(ValueError):
        dynamic_positions([source])
    with pytest.raises(ValueError, match="CAPACITY"):
        dynamic_positions(entity(f"person_{i}") for i in range(129))
    with pytest.raises(ValueError, match="CAPACITY"):
        dynamic_positions(entity("static") for _ in range(65_537))
    with pytest.raises(ValueError, match="GEOMETRY"):
        dynamic_observation(
            "person_unknown", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), {}, velocity_known=True
        )


# 功能：
#   核对实际慢速诊断分支调用与快速真值相同的几何构造器，不保留另一份行人模板。
# 输入：
#   无；读取真实适配器源码中的障碍构造表达式。
# 输出：
#   None：不返回业务数据。
def test_adapter_uses_same_declared_dynamic_geometry():
    tree = ast.parse(Path(gazebo_adapter.__file__).read_text(encoding="utf-8"))
    process = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "process_pose"
    )
    calls = [
        node
        for node in ast.walk(process)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert not any(node.func.id == "DynamicObstacleObservation" for node in calls)
    call = next(node for node in calls if node.func.id == "dynamic_observation")
    scope = dict(
        dynamic_observation=dynamic_observation,
        name="dronedream_dynamic_cart",
        root_position=(1.0, 2.0, 9.0),
        obstacle_velocity=(0.0, 0.0, 0.0),
        dynamic_geometry={"dronedream_dynamic_cart": 0.6},
        history=(0, (1.0, 2.0, 9.0)),
    )
    row = eval(compile(ast.Expression(call), "adapter-dynamic-shape", "eval"), scope)
    assert row.radius_m == 0.6 and row.height_m == 1.2 and row.position_m.z == 9.0
