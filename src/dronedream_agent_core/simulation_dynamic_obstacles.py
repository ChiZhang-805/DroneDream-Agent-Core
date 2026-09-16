"""Declared collision bounds for simulation witnesses; never camera perception."""

from __future__ import annotations

import math
import re
from pathlib import Path

from .contracts import DynamicObstacleObservation, Vector3
from .plugin_files import read_plugin_file
from .training.geometry_inputs import finite_metric, metric_vector
from .xml_values import parse_xml


# 功能：
#   识别项目约定的动态实体根名称，排除链接及有作用域的子节点。
# 输入：
#   name：原生位姿中的实体名。
# 输出：
#   matched：名称属于受监测动态实体时为 True。
def dynamic_model_name(name: str) -> bool:
    matched = (
        isinstance(name, str)
        and "::" not in name
        and name.casefold().startswith(
            ("dronedream_dynamic_", "person_", "pedestrian_", "vehicle_dynamic_")
        )
    )
    return matched


# 功能：
#   校验可绑定的动态实体身份，输出与观测契约一致的稳定标识。
# 输入：
#   name：原始实体名。
# 输出：
#   identity：用于动态障碍观测的归一标识。
def dynamic_identity(name):
    if not dynamic_model_name(name) or len(name) > 256:
        raise ValueError("OUTCOME_DYNAMIC_IDENTITY_INVALID")
    identity = re.sub(r"[^a-z0-9._-]+", "-", name.casefold()).strip("-.")
    return identity


# 功能：
#   复制有限动态位置，拒绝原始名或归一名冲突，避免字典覆盖导致障碍消失。
# 输入：
#   poses：同一有界位姿帧中的实体序列。
# 输出：
#   positions：原始实体名到不可变三维位置的映射。
def dynamic_positions(poses):
    positions, identities = {}, set()
    for index, pose in enumerate(poses):
        if index >= 65_536:
            raise ValueError("OUTCOME_POSE_ENTITY_CAPACITY")
        name = pose.name
        if not dynamic_model_name(name):
            continue
        identity = dynamic_identity(name)
        if name in positions or identity in identities:
            raise ValueError("OUTCOME_DYNAMIC_IDENTITY_AMBIGUOUS")
        if len(positions) >= 128:
            raise ValueError("OUTCOME_DYNAMIC_ENTITY_CAPACITY")
        identities.add(identity)
        positions[name] = metric_vector([pose.position.x, pose.position.y, pose.position.z])
    return positions


# 功能：
#   从唯一子元素读取固定数量的有限数值，不接受重复字段或缺失值。
# 输入：
#   parent：包含度量字段的 XML 节点。
#   tag：目标子元素名。
#   count：要求的数值数量。
# 输出：
#   values：通过类型、数量和有限性检查的数值元组。
def _numbers(parent, tag, count):
    nodes = parent.findall(tag)
    if len(nodes) != 1:
        raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_FIELD_INVALID")
    values = tuple(float(item) for item in (nodes[0].text or "").split())
    if len(values) != count or not all(finite_metric(item) for item in values):
        raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_FIELD_INVALID")
    return values


# 功能：
#   提取相对直属父节点的平移距离上界，不猜测命名坐标系或不同姿态编码。
# 输入：
#   node：链接或碰撞体节点。
# 输出：
#   distance：该节点原点相对父原点的距离，米。
def _translation_bound(node):
    poses = node.findall("pose")
    if not poses:
        return 0.0
    if len(poses) != 1 or poses[0].attrib:
        raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_FRAME_UNSUPPORTED")
    values = _numbers(node, "pose", 6)
    distance = math.hypot(*values[:3])
    return distance


# 功能：
#   按实际基本碰撞形状计算包围球半径，拒绝未解析的网格及多义几何。
# 输入：
#   collision：SDF 碰撞节点。
# 输出：
#   radius：以碰撞原点为中心、覆盖形状全部旋转的半径，米。
def _shape_radius(collision):
    geometries = collision.findall("geometry")
    if len(geometries) != 1 or len(geometries[0]) != 1:
        raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_SHAPE_INVALID")
    shape = geometries[0][0]
    if shape.tag == "box":
        dimensions = _numbers(shape, "size", 3)
        radius = math.hypot(*(item / 2 for item in dimensions))
    elif shape.tag == "sphere":
        dimensions = _numbers(shape, "radius", 1)
        radius = dimensions[0]
    elif shape.tag == "cylinder":
        dimensions = (*_numbers(shape, "radius", 1), *_numbers(shape, "length", 1))
        radius = math.hypot(dimensions[0], dimensions[1] / 2)
    else:
        raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_SHAPE_UNSUPPORTED")
    if not all(item > 0 for item in dimensions) or not finite_metric(radius):
        raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_DIMENSION_INVALID")
    return radius


# 功能：
#   1. 从当前世界的有界原始字节读取动态实体真实碰撞尺寸，不沿用固定行人模板。
#   2. 用模型根原点包围球覆盖单个刚性链接的全部旋转；外包圆柱只会偏保守。
#   3. 不猜测关节、嵌套模型、引用网格和相对命名坐标系，无法证明包络即拒绝。
# 输入：
#   world_sdf：将用于当前仿真的世界文件路径。
# 输出：
#   radii：动态实体原始名称到模型根原点包围球半径的映射。
def load_dynamic_geometry(world_sdf: Path):
    limit = 64 * 1024 * 1024
    raw = read_plugin_file(Path(world_sdf).absolute(), limit=limit)
    root = parse_xml(raw, maximum_bytes=limit, maximum_elements=1_000_000)
    worlds = root.findall("world")
    if root.tag != "sdf" or len(worlds) != 1:
        raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_WORLD_INVALID")
    radii, identities = {}, set()
    for model in worlds[0]:
        name = model.get("name")
        if not dynamic_model_name(name):
            continue
        identity = dynamic_identity(name)
        if name in radii or identity in identities:
            raise ValueError("OUTCOME_DYNAMIC_IDENTITY_AMBIGUOUS")
        if len(radii) >= 128:
            raise ValueError("OUTCOME_DYNAMIC_ENTITY_CAPACITY")
        if model.tag != "model" or any(
            model.findall(tag) for tag in ("joint", "model", "include", "frame")
        ):
            raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_MODEL_UNSUPPORTED")
        links = model.findall("link")
        if len(links) != 1:
            raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_LINK_INVALID")
        radius, collision_count = 0.0, 0
        for link in links:
            offset = _translation_bound(link)
            for collision in link.findall("collision"):
                collision_count += 1
                if collision_count > 512:
                    raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_COLLISION_CAPACITY")
                # 三角不等式不依赖机体朝向；把局部旋转误当世界固定朝向会低估包络。
                radius = max(
                    radius, offset + _translation_bound(collision) + _shape_radius(collision)
                )
        if not finite_metric(radius) or not 0 < radius <= 20:
            raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_BOUND_INVALID")
        identities.add(identity)
        radii[name] = radius
    return radii


# 功能：
#   由仿真纳秒差计算物理速度，首次无历史只提供占位零值，由调用方标记未就绪。
# 输入：
#   position：当前三维位置，米。
#   previous：前次仿真纳秒时间与位置；没有历史时为 None。
#   stamp：当前仿真纳秒时间。
# 输出：
#   velocity：有限三轴速度，米每仿真秒。
def simulation_velocity(position, previous, stamp):
    position = metric_vector(position)
    if type(stamp) is not int or not 0 <= stamp <= 2**63 - 1:
        raise ValueError("OUTCOME_SIMULATION_CLOCK_INVALID")
    if previous is None:
        return (0.0, 0.0, 0.0)
    old_stamp, old_position = previous
    old_position = metric_vector(old_position)
    if type(old_stamp) is not int or not 0 <= old_stamp < stamp:
        raise ValueError("OUTCOME_SIMULATION_CLOCK_REGRESSED")
    dt = (stamp - old_stamp) / 1e9
    velocity = metric_vector(
        tuple((p - old) / dt for p, old in zip(position, old_position, strict=True))
    )
    return velocity


# 功能：
#   生成基于当前碰撞包络的仿真障碍观测，不把未知尺寸或首次零速度当作精确真值。
# 输入：
#   name：当前实体原始名称。
#   position：模型根原点在世界中的位置，米。
#   velocity：模型根速度，米每仿真秒。
#   geometry：从当前世界提取的名称到包围球半径映射。
#   velocity_known：是否有连续两帧可计算速度。
# 输出：
#   observation：以模型根为中心的保守圆柱包络观测。
def dynamic_observation(name, position, velocity, geometry, *, velocity_known):
    identity = dynamic_identity(name)
    radius = geometry.get(name)
    if not finite_metric(radius) or not 0 < radius <= 20:
        raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_MISSING_OR_INVALID")
    position, velocity = metric_vector(position), metric_vector(velocity)
    if type(velocity_known) is not bool:
        raise ValueError("OUTCOME_DYNAMIC_VELOCITY_VALIDITY_INVALID")
    observation = DynamicObstacleObservation(
        obstacle_id=identity,
        position_m=Vector3(x=position[0], y=position[1], z=position[2]),
        velocity_mps=Vector3(x=velocity[0], y=velocity[1], z=velocity[2]),
        radius_m=radius,
        height_m=2 * radius,
        confidence=1.0 if velocity_known else 0.0,
        age_seconds=0.0,
    )
    return observation


# 功能：
#   由同一主机时基计算真实帧年龄，拒绝非有限时钟及处理早于接收的情况。
# 输入：
#   received：原始单调接收时刻，秒。
#   processed：当前单调处理时刻，秒。
# 输出：
#   age：实际经过的非负秒数，不用截零掩盖时间倒退。
def witness_age(received, processed):
    if (
        not finite_metric(received)
        or not finite_metric(processed)
        or not 0 <= received <= processed
    ):
        raise ValueError("OUTCOME_PROCESSING_CLOCK_INVALID")
    age = processed - received
    return age
