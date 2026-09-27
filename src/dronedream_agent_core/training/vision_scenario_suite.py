"""Reproducible, independently laid-out camera training scenes; no flight commands."""

from __future__ import annotations

import hashlib
import math
import random
import xml.etree.ElementTree as ET

from dronedream_agent_core.local_vision_training import LOCAL_VISION_SEMANTIC_CLASSES

SCHEMA = "dronedream.vision-scenario-suite.v1"
PERSON_SHA256 = "bc20dd2cd005d70a35627e38cb52d4c84f3d66034b196efa28e11b10b3295c15"


# 功能：
#   将米制观察目标转换为 Gazebo 相机前向轴的单位四元数，同时保留小幅横滚。
# 输入：
#   position、target、roll：相机位置、观察目标和弧度横滚角。
# 输出：
#   quaternion：按 w、x、y、z 排列的单位四元数。
def look_at(position, target, roll=0.0):
    if (len(position) != 3 or len(target) != 3
            or not all(math.isfinite(v) for v in (*position, *target, roll))):
        raise ValueError("VISION_SCENARIO_CAMERA_INVALID")
    dx, dy, dz = (b - a for a, b in zip(position, target, strict=True))
    if math.sqrt(dx * dx + dy * dy + dz * dz) < 0.1:
        raise ValueError("VISION_SCENARIO_TARGET_TOO_CLOSE")
    yaw, pitch = math.atan2(dy, dx), -math.atan2(dz, math.hypot(dx, dy))
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    quaternion = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                  cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
    return quaternion


# 功能：
#   按不依赖语义类别的分布生成材质，避免模型只通过固定颜色猜测标签。
# 输入：
#   rng：本场景独立的随机数发生器。
# 输出：
#   color：有界的 RGBA 材质分量。
def surface_color(rng):
    value = rng.uniform(0.3, 0.82)
    # 实际学校地图以低饱和建筑材质为主；不把随机鲜艳颜色当作泛化能力的替代品。
    color = [round(value + rng.uniform(-0.04, 0.04), 4) for _ in range(3)] + [1.0]
    return color


class SceneBuilder:
    """Explicit static geometry, labels and conservative camera exclusion volumes."""

    # 功能：
    #   初始化静态场景并独立随机化环境光、天空和定向光，不引用软件运行地图。
    # 输入：
    #   seed：整布局的固定随机种子。
    # 输出：
    #   None：建立 XML、标签表与相机禁入体积。
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.root = ET.Element("sdf", version="1.9")
        self.world = ET.SubElement(self.root, "world", name=f"campus_layout_{seed}")
        self.labels, self.volumes = {}, []
        scene = ET.SubElement(self.world, "scene")
        ambient = self.rng.uniform(0.25, 0.7)
        ET.SubElement(scene, "ambient").text = f"{ambient} {ambient} {ambient} 1"
        sky = self.rng.uniform(0.5, 0.8)
        ET.SubElement(scene, "background").text = f"{sky * 0.85} {sky * 0.94} {sky} 1"
        light = ET.SubElement(self.world, "light", name="sun", type="directional")
        ET.SubElement(light, "direction").text = (
            f"{self.rng.uniform(-0.8, 0.8)} {self.rng.uniform(-0.8, 0.8)} -1")
        ET.SubElement(light, "diffuse").text = "0.8 0.8 0.8 1"
        ET.SubElement(light, "cast_shadows").text = "true"

    # 功能：
    #   创建具有显式标签的可见面；名称冲突和非法类别直接报错。
    # 输入：
    #   name、class_id、position：唯一名称、语义类别和中心位置。
    # 输出：
    #   visual：等待填入具体几何的 XML 节点。
    def visual(self, name, class_id, position):
        key = f"{name}::link::visual"
        if key in self.labels or type(class_id) is not int or not 0 <= class_id < 8:
            raise ValueError("VISION_SCENARIO_OBJECT_INVALID")
        self.labels[key] = class_id
        model = ET.SubElement(self.world, "model", name=name)
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = " ".join(map(str, [*position, 0, 0, 0]))
        link = ET.SubElement(model, "link", name="link")
        visual = ET.SubElement(link, "visual", name="visual")
        return visual

    # 功能：
    #   生成带随机材质的实体盒体，并登记其几何体积以排除穿墙相机。
    # 输入：
    #   name、class_id、position、size、glass：对象身份、中心、尺寸及透明玻璃标记。
    # 输出：
    #   None：添加已标注可见面和禁入体积。
    def box(self, name, class_id, position, size, glass=False):
        if any(not math.isfinite(v) for v in (*position, *size)) or min(size) <= 0:
            raise ValueError("VISION_SCENARIO_BOX_INVALID")
        visual = self.visual(name, class_id, position)
        ET.SubElement(ET.SubElement(ET.SubElement(visual, "geometry"), "box"), "size").text = (
            " ".join(map(str, size)))
        material = ET.SubElement(visual, "material")
        color = surface_color(self.rng)
        for tag in ("ambient", "diffuse"):
            ET.SubElement(material, tag).text = " ".join(map(str, color))
        if glass:
            ET.SubElement(visual, "transparency").text = str(self.rng.uniform(0.35, 0.7))
            ET.SubElement(material, "specular").text = "0.8 0.8 0.8 1"
        self.volumes.append((tuple(position), tuple(size)))

    # 功能：
    #   放入已核对来源的人形静态网格，随机化朝向但不宣称动态行为或身份多样性。
    # 输入：
    #   name、x、y：唯一名称及地面位置。
    # 输出：
    #   None：添加行人可见面和保守的相机禁入范围。
    def person(self, name, x, y):
        visual = self.visual(name, 5, [x, y, 0.02])
        ET.SubElement(visual, "pose").text = f"0 0 0 0.04 0 {self.rng.uniform(-math.pi, math.pi)}"
        mesh = ET.SubElement(ET.SubElement(visual, "geometry"), "mesh")
        ET.SubElement(mesh, "uri").text = "person/meshes/standing.dae"
        self.volumes.append(((x, y, 0.98), (0.9, 0.9, 1.96)))

    # 功能：
    #   检查候选光学中心是否具有实体间隙，不能靠相机穿墙获得不可能的训练视角。
    # 输入：
    #   position、margin：相机位置及各方向保守间距。
    # 输出：
    #   clear：位于全部实体之外且离地足够高时为 True。
    def camera_clear(self, position, margin=0.15):
        clear = position[2] >= 0.4 and all(not all(
            abs(p - c) <= s / 2 + margin for p, c, s in zip(position, center, size, strict=True)
        ) for center, size in self.volumes)
        return clear


# 功能：
#   建造独立的房间、门洞、玻璃窗、楼梯和取件庭院，几何尺寸及物体相对位置均可变化。
# 输入：
#   builder：具有独立种子的场景构建器。
# 输出：
#   anchors、bounds：八类观察目标和室内边界。
def populate_campus(builder):
    rng, box = builder.rng, builder.box
    length, half_width, height = rng.uniform(9, 13), rng.uniform(4, 6), rng.uniform(3.4, 4.8)
    door_width, door_height = rng.uniform(1.2, 2.4), rng.uniform(2.3, 3.1)
    box("ground", 1, [10, 0, -0.1], [60, 50, 0.2])
    box("room-floor", 1, [length / 2, 0, 0.025], [length, 2 * half_width, 0.05])
    box("roof", 2, [length / 2, 0, height], [length, 2 * half_width, 0.2])
    box("back", 2, [0, 0, height / 2], [0.2, 2 * half_width, height])
    box("north", 2, [length / 2, half_width, height / 2], [length, 0.2, height])
    # 玻璃嵌在真实窗洞中；不把玻璃叠在不透明整面墙上。
    window_x, window_width = length * 0.42, rng.uniform(2.2, 3.6)
    sill, top = 0.8, height - 0.7
    for suffix, x0, x1 in (("left", 0, window_x - window_width / 2),
                            ("right", window_x + window_width / 2, length)):
        box("south-" + suffix, 2, [(x0 + x1) / 2, -half_width, height / 2],
            [x1 - x0, 0.2, height])
    box("sill", 2, [window_x, -half_width, sill / 2], [window_width, 0.2, sill])
    box("lintel", 2, [window_x, -half_width, (top + height) / 2],
        [window_width, 0.2, height - top])
    box("glass", 6, [window_x, -half_width, (sill + top) / 2],
        [window_width, 0.06, top - sill], glass=True)
    for sign in (-1, 1):
        width = half_width - door_width / 2
        box(f"front-{sign}", 2, [length, sign * (half_width + door_width / 2) / 2, height / 2],
            [0.2, width, height])
        box(f"door-post-{sign}", 3,
            [length - 0.05, sign * (door_width / 2 + 0.075), door_height / 2],
            [0.3, 0.15, door_height])
    box("door-header", 3, [length, 0, door_height + 0.1], [0.3, door_width + 0.3, 0.2])
    box("front-upper", 2, [length, 0, (door_height + 0.2 + height) / 2],
        [0.2, door_width, height - door_height - 0.2])
    # 楼梯由真实逐级几何构成，标签不能只画在平面或在图像上叠字。
    steps, run, rise = rng.randint(5, 10), rng.uniform(0.24, 0.42), rng.uniform(0.13, 0.22)
    stair_x, stair_y = length * 0.54, half_width - 1.35
    for index in range(steps):
        z = (index + 1) * rise
        box(f"stair-{index}", 4, [stair_x + index * run, stair_y, z / 2], [run, 1.8, z])
    pickup_x, pickup_y = length + rng.uniform(6, 10), rng.uniform(-4, -2)
    box("pickup-counter", 2, [pickup_x, pickup_y, 0.5], [rng.uniform(1.8, 3), 1.2, 1])
    box("pickup-marker", 7, [pickup_x, pickup_y, 1.025], [rng.uniform(0.5, 1.1), 0.8, 0.05])
    box("pickup-canopy", 2, [pickup_x, pickup_y, 3.2], [4.2, 2.8, 0.16])
    for sign in (-1, 1):
        box(f"canopy-post-{sign}", 2, [pickup_x + sign * 1.9, pickup_y + 1.2, 1.6],
            [0.12, 0.12, 3.2])
    person_x, person_y = length + rng.uniform(4, 7), rng.uniform(2.2, 4)
    builder.person("person-outdoor", person_x, person_y)
    builder.person("person-indoor", length * 0.26, half_width * 0.35)
    for index in range(10):
        x, y = rng.uniform(length + 3, length + 15), rng.choice((-1, 1)) * rng.uniform(6, 10)
        z = rng.uniform(0.4, 2.6)
        box(f"yard-obstacle-{index}", 2, [x, y, z / 2],
            [rng.uniform(0.3, 1.8), rng.uniform(0.3, 1.8), z])
    box("desk", 2, [2.4, -half_width + 1.2, 0.42], [1.5, 0.8, 0.84])
    anchors = [
        ("door", [length, 0, 1.5], [length - 3.7, 0, 1.6], "indoor"),
        ("stairs", [stair_x + steps * run / 2, stair_y, steps * rise / 2],
         [stair_x - 1.7, stair_y - 1.9, 1.8], "indoor"),
        ("window", [window_x, -half_width, (sill + top) / 2],
         [window_x, -half_width + 2.8, 1.8], "indoor"),
        ("person-room", [length * 0.26, half_width * 0.35, 0.9],
         [length * 0.26 + 2.6, half_width * 0.35, 1.3], "indoor"),
        ("pickup", [pickup_x, pickup_y, 1.02], [pickup_x - 2.8, pickup_y - 2.0, 2.0], "outdoor"),
        ("person-yard", [person_x, person_y, 0.9],
         [person_x + 2.7, person_y - 1.0, 1.4], "outdoor"),
        ("entrance", [length, 0, 1.8], [length + 4.2, 0, 1.8], "outdoor"),
        ("yard", [length + 10, 6, 1.2], [length + 2.5, 4.5, 2.3], "outdoor"),
    ]
    return anchors, (length, half_width, height)


# 功能：
#   在同一整布局分组内生成不同位置、角度和高度，质量扰动不改变数据所属集合。
# 输入：
#   builder、anchors、bounds、seed、split、views_per_anchor：场景、观察目标、边界及采集数量。
# 输出：
#   views：经过实体间隙和室内外归属校验的独立相机视角。
def sample_views(builder, anchors, bounds, seed, split, views_per_anchor):
    rng, views = builder.rng, []
    length, half_width, height = bounds
    for name, target, center, setting in anchors:
        for index in range(views_per_anchor):
            for _ in range(500):
                position = [center[0] + rng.uniform(-1.1, 1.1), center[1] + rng.uniform(-0.9, 0.9),
                            max(0.5, min(height - 0.4, center[2] + rng.uniform(-0.6, 0.7)))]
                indoor = 0.2 < position[0] < length - 0.2 and abs(position[1]) < half_width - 0.2
                if builder.camera_clear(position) and indoor == (setting == "indoor"):
                    break
            else:
                raise ValueError("VISION_SCENARIO_NO_VALID_CAMERA")
            aim = [target[0] + rng.uniform(-0.25, 0.25), target[1] + rng.uniform(-0.25, 0.25),
                   target[2] + rng.uniform(-0.15, 0.15)]
            # 每个扰动样本仍对应新渲染的视角；原始帧也随来源保存。
            mode, corruption = index % 16, {}
            if mode in (1, 9):
                corruption = {"exposure_multiplier": rng.uniform(0.035, 0.07)}
            elif mode in (2, 10):
                corruption = {"exposure_multiplier": rng.uniform(3.5, 4.0)}
            elif mode in (3, 7, 11):
                corruption = {"blur_sigma_px": rng.uniform(1.1, 3.8)}
            elif mode in (4, 8, 12):
                left, top = rng.uniform(0.05, 0.5), rng.uniform(0.05, 0.45)
                corruption = {"occlusion_rect": [left, top, left + rng.uniform(0.15, 0.4),
                                                  top + rng.uniform(0.15, 0.4)]}
            views.append({"view_id": f"{name}-{index:03d}",
                          "scene_group_id": f"campus-layout-{seed}",
                          "split": split, "position_m": position,
                          "orientation_wxyz": look_at(position, aim, rng.uniform(-0.13, 0.13)),
                          "setting": setting, "corruption": corruption})
    return views


# 功能：
#   构造有界、可重现的独立几何布局与分组扫描计划，并明确单一人形和静态监督的限制。
# 输入：
#   seed、split、views_per_anchor：整布局种子、数据划分以及每个观察区域的帧数。
# 输出：
#   world_bytes、labels、plan、receipt：源世界、完整标签、视角计划和布局来源回执。
def build_scenario(seed, split, views_per_anchor=16):
    if (type(seed) is not int or not 0 <= seed < 2**31
            or split not in ("training", "validation", "test")
            or type(views_per_anchor) is not int or not 2 <= views_per_anchor <= 128):
        raise ValueError("VISION_SCENARIO_REQUEST_INVALID")
    builder = SceneBuilder(seed)
    anchors, bounds = populate_campus(builder)
    views = sample_views(builder, anchors, bounds, seed, split, views_per_anchor)
    world_bytes = ET.tostring(builder.root, encoding="utf-8", xml_declaration=True)
    world_sha256 = hashlib.sha256(world_bytes).hexdigest()
    labels = {"schema": "dronedream.vision-world-labels.v1", "world_sha256": world_sha256,
              "classes": list(LOCAL_VISION_SEMANTIC_CLASSES), "visuals": builder.labels}
    plan = {"schema": "dronedream.vision-view-plan.v1", "views": views}
    receipt = {"schema": SCHEMA, "seed": seed, "split": split, "world_sha256": world_sha256,
               "bounds": bounds, "anchors": anchors, "views": len(views),
               "scene_group_id": f"campus-layout-{seed}", "independent_geometry_layout": True,
               "person_identities": 1, "person_pose": "static-natural-standing",
               "photorealism_proven": False, "physical_flight_evidence": False,
               "semantic_labels_are_visibility_not_free_space": True}
    return world_bytes, labels, plan, receipt
