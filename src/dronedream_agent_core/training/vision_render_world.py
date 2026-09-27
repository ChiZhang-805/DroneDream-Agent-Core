"""Derive a labelled camera-only Gazebo world; never command a flying vehicle."""

from __future__ import annotations

import copy
import hashlib
import math
import posixpath
import xml.etree.ElementTree as ET
from pathlib import Path

from dronedream_agent_core.geometry_motion_fixture import POSE_TOPIC, RIG_NAME
from dronedream_agent_core.local_vision_training import LOCAL_VISION_SEMANTIC_CLASSES
from dronedream_agent_core.plugin_files import portable_plugin_path, read_plugin_file
from dronedream_agent_core.xml_values import parse_xml

RGB_TOPIC = "/dronedream/vision_dataset/rgb"
SEMANTIC_TOPIC = "/dronedream/vision_dataset/semantic"


# 功能：
#   核查 COLLADA 内部图像引用，把间接纹理一并纳入采集前后摘要，拒绝目录外引用。
# 输入：
#   relative、raw、source_root：网格相对路径、原始字节和资源根目录。
# 输出：
#   textures：真实普通纹理文件的相对路径到摘要映射。
def collada_texture_hashes(relative, raw, source_root):
    if Path(relative).suffix.lower() != ".dae":
        return {}
    tree = parse_xml(raw, maximum_bytes=32 * 1024**2)
    textures = {}
    for node in tree.findall(".//{*}library_images/{*}image/{*}init_from"):
        reference = (node.text or "").strip()
        if not reference or "\\" in reference or ":" in reference or reference.startswith("/"):
            raise ValueError("VISION_RENDER_TEXTURE_REFERENCE_INVALID")
        normalized = posixpath.normpath(posixpath.join(posixpath.dirname(relative), reference))
        safe = portable_plugin_path(normalized)
        if Path(safe).suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise ValueError("VISION_RENDER_TEXTURE_FORMAT_UNSUPPORTED")
        payload = read_plugin_file(source_root / safe, limit=32 * 1024**2)
        textures[normalized] = hashlib.sha256(payload).hexdigest()
    return textures


# 功能：
#   将派生世界中的相对网格和贴图绑定回已验证源目录，防止换目录后静默丢失材质。
# 输入：
#   world_bytes、source_root：派生 SDF 与原始地图资源目录。
# 输出：
#   rebased、hashes：引用明确普通文件的世界字节及实际资源摘要。
def bind_render_resources(world_bytes: bytes, source_root: Path):
    root = parse_xml(world_bytes, maximum_bytes=32 * 1024**2)
    tags = {"uri", "albedo_map", "normal_map", "roughness_map", "metalness_map",
            "environment_map", "emissive_map", "light_map", "glossiness_map", "specular_map"}
    hashes = {}
    for node in root.iter():
        if node.tag not in tags:
            continue
        relative = portable_plugin_path(node.text or "")
        path = source_root.absolute() / relative
        raw = read_plugin_file(path, limit=32 * 1024**2)
        hashes[relative.as_posix() if isinstance(relative, Path) else str(relative)] = (
            hashlib.sha256(raw).hexdigest())
        hashes.update(collada_texture_hashes(str(relative).replace("\\", "/"), raw,
                                             source_root.absolute()))
        node.text = path.as_posix()
    rebased = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return rebased, hashes


# 功能：
#   递归索引显式静态模型的可见几何，拒绝重复名称与外部包含，不猜测未知对象的类别。
# 输入：
#   parent、prefix：地图模型树及父级命名空间。
# 输出：
#   visuals：完整 model::link::visual 名称到 XML 节点的映射。
def index_static_visuals(parent, prefix=""):
    visuals, names = {}, set()
    for model in parent.findall("model"):
        name = model.get("name", "")
        if not name or "::" in name or name in names:
            raise ValueError("VISION_WORLD_MODEL_NAME_AMBIGUOUS")
        names.add(name)
        if model.findtext("static", "false").strip() not in ("true", "1"):
            raise ValueError("VISION_RENDER_WORLD_NOT_STATIC")
        path = prefix + name + "::"
        link_names = set()
        for link in model.findall("link"):
            link_name = link.get("name", "")
            if not link_name or "::" in link_name or link_name in link_names:
                raise ValueError("VISION_WORLD_LINK_NAME_AMBIGUOUS")
            link_names.add(link_name)
            for visual in link.findall("visual"):
                visual_name = visual.get("name", "")
                key = path + link_name + "::" + visual_name
                if not visual_name or "::" in visual_name or key in visuals:
                    raise ValueError("VISION_WORLD_VISUAL_NAME_AMBIGUOUS")
                visuals[key] = visual
        nested = index_static_visuals(model, path)
        if visuals.keys() & nested.keys():
            raise ValueError("VISION_WORLD_VISUAL_NAME_AMBIGUOUS")
        visuals.update(nested)
    return visuals


# 功能：
#   1. 逐个可见几何应用显式语义标签；未标注、重复或错误版本地图不能生成训练世界。
#   2. 安装同位姿、同内参的 RGB 与分割相机，仅使用固定 Gazebo 系统，不安装飞控。
# 输入：
#   world_bytes、labels：原始静态 SDF 与绑定其摘要的完整类别分配表。
#   camera_bytes：唯一前向 RGB 相机的 SDF；必须为无畸变、无嵌入执行插件的配置。
# 输出：
#   derived、metadata：派生世界字节及来源/相机身份，不覆盖原地图。
def build_labelled_render_world(world_bytes: bytes, labels: dict, camera_bytes: bytes):
    root = parse_xml(world_bytes, maximum_bytes=32 * 1024**2)
    camera_root = parse_xml(camera_bytes, maximum_bytes=1024**2)
    if len(root.findall("world")) != 1:
        raise ValueError("VISION_RENDER_WORLD_AMBIGUOUS")
    world = root.find("world")
    if any(world.findall(".//" + tag) for tag in ("plugin", "sensor", "actor", "include")):
        raise ValueError("VISION_RENDER_WORLD_HAS_EXECUTABLE_OR_EXTERNAL_CONTENT")
    if world.find(f"model[@name='{RIG_NAME}']") is not None:
        raise ValueError("VISION_RENDER_RIG_NAME_COLLISION")
    source_sha256 = hashlib.sha256(world_bytes).hexdigest()
    if (type(labels) is not dict or set(labels) != {"schema", "world_sha256", "classes", "visuals"}
            or labels["schema"] != "dronedream.vision-world-labels.v1"
            or labels["world_sha256"] != source_sha256
            or labels["classes"] != list(LOCAL_VISION_SEMANTIC_CLASSES)
            or type(labels["visuals"]) is not dict):
        raise ValueError("VISION_RENDER_LABEL_BINDING_INVALID")
    visuals = index_static_visuals(world)
    if not visuals or set(visuals) != set(labels["visuals"]):
        raise ValueError("VISION_RENDER_LABEL_COVERAGE_INCOMPLETE")
    for key, visual in visuals.items():
        class_id = labels["visuals"][key]
        if type(class_id) is not int or not 0 <= class_id < len(LOCAL_VISION_SEMANTIC_CLASSES):
            raise ValueError("VISION_RENDER_LABEL_CLASS_INVALID")
        plugin = ET.SubElement(visual, "plugin", filename="gz-sim-label-system",
                               name="gz::sim::systems::Label")
        ET.SubElement(plugin, "label").text = str(class_id)
    cameras = [sensor for sensor in camera_root.iter("sensor") if sensor.get("type") == "camera"]
    if len(cameras) != 1:
        raise ValueError("VISION_RENDER_FORWARD_CAMERA_AMBIGUOUS")
    sensor = cameras[0]
    if (sensor.findall(".//plugin") or sensor.findall(".//include")
            or len(sensor.findall("camera")) != 1):
        raise ValueError("VISION_RENDER_CAMERA_EXECUTION_OR_STRUCTURE_INVALID")
    optical = sensor.find("camera")
    if any(optical.find(tag) is not None
           for tag in ("lens", "distortion", "save", "intrinsics", "projection")):
        raise ValueError("VISION_RENDER_CAMERA_PROJECTION_UNSUPPORTED")
    fov = float(optical.findtext("horizontal_fov"))
    near, far = float(optical.findtext("clip/near")), float(optical.findtext("clip/far"))
    if (not all(math.isfinite(value) for value in (fov, near, far))
            or not 0.1 <= fov <= 2.8 or not 0 < near < far <= 1000):
        raise ValueError("VISION_RENDER_CAMERA_INTRINSICS_INVALID")
    width, height = int(optical.findtext("image/width")), int(optical.findtext("image/height"))
    if not 1 <= width <= 1280 or not 1 <= height <= 720 or width * height > 1280 * 720:
        raise ValueError("VISION_RENDER_CAMERA_PIXEL_BUDGET")
    world.set("name", "vision_dataset_world")
    for name, library in (("Physics", "physics"), ("UserCommands", "user-commands"),
                          ("SceneBroadcaster", "scene-broadcaster"), ("Sensors", "sensors")):
        plugin = ET.SubElement(world, "plugin", name=f"gz::sim::systems::{name}",
                               filename=f"gz-sim-{library}-system")
        if name == "Sensors":
            ET.SubElement(plugin, "render_engine").text = "ogre2"
    rig = ET.SubElement(world, "model", name=RIG_NAME, canonical_link="camera_link")
    ET.SubElement(rig, "static").text = "true"
    link = ET.SubElement(rig, "link", name="camera_link")
    for name, kind, topic in (("rgb", "camera", RGB_TOPIC),
                               ("semantic", "segmentation", SEMANTIC_TOPIC)):
        derived_sensor = copy.deepcopy(sensor)
        derived_sensor.set("name", name)
        derived_sensor.set("type", kind)
        for tag in ("pose", "topic", "update_rate", "visualize"):
            for old in derived_sensor.findall(tag):
                derived_sensor.remove(old)
        # 扫描计划描述光学坐标原点，而不是机体；两相机完全同位，不套用旧机体偏移。
        for tag, value in (("pose", "0 0 0 0 0 0"), ("topic", topic),
                           ("update_rate", "10"), ("visualize", "false")):
            ET.SubElement(derived_sensor, tag).text = value
        image_format = derived_sensor.find("camera/image/format")
        if image_format is not None:
            image_format.text = "R8G8B8"
        if kind == "segmentation":
            ET.SubElement(derived_sensor.find("camera"), "segmentation_type").text = "semantic"
        link.append(derived_sensor)
    publisher = ET.SubElement(rig, "plugin", name="gz::sim::systems::PosePublisher",
                               filename="gz-sim-pose-publisher-system")
    for tag, value in (("publish_model_pose", "true"), ("publish_link_pose", "false"),
                       ("use_pose_vector_msg", "false"), ("update_frequency", "100"),
                       ("topic", POSE_TOPIC)):
        ET.SubElement(publisher, tag).text = value
    derived = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    metadata = {"source_kind": "rendered-view", "world_sha256": source_sha256,
                "camera_sha256": hashlib.sha256(camera_bytes).hexdigest(),
                "derived_sha256": hashlib.sha256(derived).hexdigest(),
                "labelled_visual_count": len(visuals), "width": width, "height": height,
                "physical_flight_evidence": False}
    return derived, metadata
