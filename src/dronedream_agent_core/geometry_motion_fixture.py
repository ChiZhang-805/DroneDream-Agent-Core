"""Camera-only moving test fixture, NOT an aircraft controller or training source.

The fixture uses real Gazebo depth pixels. Its commanded poses and deliberately
perturbed pose inputs test registration; they do not qualify native estimation,
flight dynamics, uncertainty, neural policies, or autonomous mission success.
"""

from __future__ import annotations

import copy
import hashlib
import math
import xml.etree.ElementTree as ET

import numpy as np

from .contracts import CalibratedRangeSensorMount
from .depth_projection import DepthProjectionCalibration, metric_depth_sample_stride
from .quaternion_geometry import rotate_vector
from .runtime_sensor_contracts import oakd_lite_depth_sensor_contract
from .simulation_sensor_frames import _rotation

RIG_NAME = "dronedream_geometry_fixture"
POSE_TOPIC = "/dronedream/geometry_fixture/pose"
DEPTH_TOPIC = "/dronedream/geometry_fixture/depth"


# 功能：
#   1. 生成局部闭合相机扫描姿态，水平主轴行程 0.8 米并包含高度及三轴转动。
#   2. 夹具指令只用于校准；实际运动必须另外观测，不据此认定飞机完成任务。
# 输入：
#   origin：扫描中心的世界坐标，单位米。
#   direction：确定水平扫描方向的三维向量，模长不决定行程。
#   phase：仿真已过时间与扫描时长之比，限制到零至一。
# 输出：
#   result：包含 position_m 与 orientation_wxyz 的夹具目标姿态。
def fixture_pose(origin, direction, phase: float) -> dict:
    origin, direction = np.asarray(origin, float), np.asarray(direction, float)
    if (origin.shape != (3,) or direction.shape != (3,)
            or not np.isfinite([origin, direction]).all()
            or max(abs(origin)) > 1e5
            or type(phase) not in (int, float) or not math.isfinite(phase)):
        raise ValueError("MOTION_FIXTURE_TRAJECTORY_INVALID")
    # hypot 避免先平方溢出；无法表示的真实模长仍拒绝，而不是静默归一化为零。
    length = math.hypot(*direction[:2])
    if not math.isfinite(length) or length < 1e-6:
        raise ValueError("MOTION_FIXTURE_TRAJECTORY_INVALID")
    forward = np.array([*direction[:2], 0.]) / length
    side = np.array([-forward[1], forward[0], 0.])
    theta = 2 * math.pi * min(1., max(0., phase))
    position = (origin + .4*(1-math.cos(theta))*forward + .12*math.sin(theta)*side
                + np.array([0., 0., .12*math.sin(2*theta)]))
    angles = np.radians([8*math.sin(2*theta), 6*math.sin(theta), 30*math.sin(theta)])
    result = {"position_m": position.tolist(), "orientation_wxyz": euler_quaternion(angles)}
    return result


# 功能：
#   将滚转、俯仰、偏航转换成 Rz(yaw)Ry(pitch)Rx(roll) 的 wxyz 姿态表示。
# 输入：
#   angles：roll、pitch、yaw，单位弧度，采用世界从局部的旋转约定。
# 输出：
#   quaternion：标量在前的四元数列表，表示局部向量到世界坐标的旋转。
def euler_quaternion(angles):
    angles = np.asarray(angles, dtype=float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("MOTION_FIXTURE_ANGLES_INVALID")
    roll, pitch, yaw = angles / 2
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    quaternion = [float(cr*cp*cy+sr*sp*sy), float(sr*cp*cy-cr*sp*sy),
                  float(cr*sp*cy+sr*cp*sy), float(cr*cp*sy-sr*sp*cy)]
    return quaternion


# 功能：
#   提取唯一深度相机的真实内参，拒绝额外执行内容和当前投影器尚不支持的覆盖配置。
# 输入：
#   camera_content：相机 SDF 原始字节，最多 1 MiB。
# 输出：
#   calibration：保留分辨率、视场角、裁剪距离及采样步长的深度校准。
def camera_calibration(camera_content: bytes) -> DepthProjectionCalibration:
    if len(camera_content) > 1024*1024:
        raise ValueError("MOTION_FIXTURE_CAMERA_TOO_LARGE")
    sensors = [s for s in ET.fromstring(camera_content).iter("sensor")
               if s.get("name") == "StereoOV7251" and s.get("type") == "depth_camera"]
    if len(sensors) != 1 or len(sensors[0].findall("camera")) != 1:
        raise ValueError("MOTION_FIXTURE_CAMERA_AMBIGUOUS")
    sensor = sensors[0]
    if any(sensor.findall(f".//{tag}") for tag in ("plugin", "include", "actor", "sensor")):
        raise ValueError("MOTION_FIXTURE_CAMERA_HAS_EXECUTABLE_SYSTEMS")
    camera = sensor.find("camera")
    if (camera.findtext("image/format") != "R_FLOAT32"
            or any(camera.find(path) is not None for path in
                   ("intrinsics", "projection", "depth_camera/clip", "lens"))):
        raise ValueError("MOTION_FIXTURE_CAMERA_PROJECTION_UNSUPPORTED")
    width, height = int(camera.findtext("image/width")), int(camera.findtext("image/height"))
    if width * height > 640 * 480:
        raise ValueError("MOTION_FIXTURE_CAMERA_BUFFER_BUDGET")
    calibration = DepthProjectionCalibration(width, height,
        float(camera.findtext("horizontal_fov")),
        float(camera.findtext("clip/near")), float(camera.findtext("clip/far")),
        sample_stride_pixels=metric_depth_sample_stride(width=width, height=height),
        no_return_mode="gazebo-far-clip")
    return calibration


# 功能：
#   按米制安装契约构造旋转矩阵，消除契约允许的四元数模长舍入误差。
# 输入：
#   mount：传感器到机体的安装外参。
# 输出：
#   rotation：三列分别为传感器坐标轴在机体坐标中的单位向量。
def _mount_rotation(mount: CalibratedRangeSensorMount) -> np.ndarray:
    rotation = np.column_stack([rotate_vector(mount.orientation_body_from_sensor, axis)
        for axis in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))])
    return rotation


# 功能：
#   将安装旋转转换为 SDF 默认弧度欧拉角，万向节锁位置采用零滚转的等价姿态。
# 输入：
#   mount：当前传感器安装外参。
# 输出：
#   angles：roll、pitch、yaw 三元组。
def _mount_angles(mount: CalibratedRangeSensorMount) -> tuple:
    rotation = _mount_rotation(mount)
    horizontal = math.hypot(rotation[0, 0], rotation[1, 0])
    pitch = math.atan2(-rotation[2, 0], horizontal)
    if horizontal > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    angles = roll, pitch, yaw
    return angles


# 功能：
#   验证深度投影量程与当前安装量程一致，避免采集夹具和离线射线转换使用不同边界。
# 输入：
#   calibration：已验证的深度投影内参及量程。
# 输出：
#   mount：与该投影匹配的当前安装契约。
def fixture_sensor_mount(calibration: DepthProjectionCalibration) -> CalibratedRangeSensorMount:
    mount = oakd_lite_depth_sensor_contract()
    if (calibration.minimum_depth_m != mount.minimum_range_m
            or calibration.maximum_depth_m != mount.maximum_range_m):
        raise ValueError("MOTION_FIXTURE_CALIBRATION_RANGE_MISMATCH")
    return mount


# 功能：
#   1. 从静态地图和真实相机定义派生无电机的隔离校准世界，只安装固定诊断系统。
#   2. 保留相机内参、噪声及更新频率，按当前安装契约设置平移与旋转外参。
# 输入：
#   world_content：静态地图 SDF 字节，最多 32 MiB。
#   camera_content：真实相机 SDF 字节，最多 1 MiB。
#   origin：相机夹具起始世界位置，单位米。
# 输出：
#   content：派生 SDF 字节，不修改原始地图与相机文件。
def build_fixture_world(world_content: bytes, camera_content: bytes, origin) -> bytes:
    if len(world_content) > 32*1024*1024 or len(camera_content) > 1024*1024:
        raise ValueError("MOTION_FIXTURE_SOURCE_TOO_LARGE")
    mount = fixture_sensor_mount(camera_calibration(camera_content))
    root, camera_root = ET.fromstring(world_content), ET.fromstring(camera_content)
    worlds = root.findall("world")
    cameras = [s for s in camera_root.iter("sensor")
               if s.get("name") == "StereoOV7251" and s.get("type") == "depth_camera"]
    if len(worlds) != 1 or len(cameras) != 1:
        raise ValueError("MOTION_FIXTURE_SOURCE_AMBIGUOUS")
    world = worlds[0]
    if (world.findall(".//include") or world.findall(".//actor")
            or world.findall(f"model[@name='{RIG_NAME}']")):
        raise ValueError("MOTION_FIXTURE_WORLD_REQUIRES_STATIC_MAP")
    if any(m.findtext("static", "false").strip() not in {"true", "1"}
           for m in world.findall("model")):
        raise ValueError("MOTION_FIXTURE_WORLD_HAS_DYNAMIC_ACTORS")
    # 地图与复制的相机都不能夹带额外执行代码，防止校准流程启动控制插件。
    if world.findall(".//plugin") or world.findall(".//sensor"):
        raise ValueError("MOTION_FIXTURE_WORLD_HAS_EXECUTABLE_SYSTEMS")
    position = fixture_pose(origin, [1, 0, 0], 0)["position_m"]
    for name, library in (("Physics", "physics"), ("UserCommands", "user-commands"),
                          ("SceneBroadcaster", "scene-broadcaster"), ("Sensors", "sensors")):
        plugin = ET.SubElement(world, "plugin", name=f"gz::sim::systems::{name}",
                               filename=f"gz-sim-{library}-system")
        if name == "Sensors":
            ET.SubElement(plugin, "render_engine").text = "ogre2"
    rig = ET.SubElement(world, "model", name=RIG_NAME, canonical_link="camera_link")
    ET.SubElement(rig, "static").text = "true"
    ET.SubElement(rig, "pose").text = " ".join(map(str, [*position, 0, 0, 0]))
    link = ET.SubElement(rig, "link", name="camera_link")
    ET.SubElement(link, "pose").text = " ".join(map(str, [
        mount.translation_body_m.x, mount.translation_body_m.y, mount.translation_body_m.z,
        *_mount_angles(mount)]))
    sensor = copy.deepcopy(cameras[0])
    for tag, value in (("pose", "0 0 0 0 0 0"), ("topic", DEPTH_TOPIC), ("visualize", "false")):
        for old in sensor.findall(tag):
            sensor.remove(old)
        ET.SubElement(sensor, tag).text = value
    link.append(sensor)
    publisher = ET.SubElement(rig, "plugin", name="gz::sim::systems::PosePublisher",
                               filename="gz-sim-pose-publisher-system")
    for tag, value in (("publish_model_pose", "true"), ("publish_link_pose", "false"),
                       ("publish_nested_model_pose", "false"), ("use_pose_vector_msg", "false"),
                       ("update_frequency", "250"), ("topic", POSE_TOPIC)):
        ET.SubElement(publisher, tag).text = value
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return content


# 功能：
#   将校准范围内的命中射线依次通过传感器安装姿态及机体姿态变换到世界坐标。
# 输入：
#   samples：有限的米制投影射线序列，最多 512 条命中射线。
#   pose：与深度图同步的机体位置及 wxyz 姿态。
# 输出：
#   result：世界坐标命中点数组及传感器发射原点，单位米。
def sensor_hits_world(samples, pose: dict):
    mount = oakd_lite_depth_sensor_contract()
    rotation = _rotation(pose["orientation_wxyz"])
    position = np.asarray(pose["position_m"], dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("MOTION_FIXTURE_POSE_INVALID")
    offset = np.array([mount.translation_body_m.x, mount.translation_body_m.y,
                       mount.translation_body_m.z])
    rays = [s for s in samples if s.hit]
    if not 1 <= len(rays) <= 512:
        raise ValueError("MOTION_FIXTURE_RAY_BUDGET_INVALID")
    directions = np.array([[s.direction_sensor.x, s.direction_sensor.y, s.direction_sensor.z]
                           for s in rays])
    lengths = np.hypot.reduce(directions, axis=1, keepdims=True)
    ranges = np.array([s.range_m for s in rays])
    if (not np.isfinite(directions).all() or not np.isfinite(ranges).all()
            or not np.isfinite(lengths).all() or np.min(lengths) <= 1e-9
            or np.min(ranges) < mount.minimum_range_m or np.max(ranges) > mount.maximum_range_m):
        raise ValueError("MOTION_FIXTURE_RAYS_INVALID")
    directions /= lengths
    local = (directions * ranges[:, None]) @ _mount_rotation(mount).T
    origin = position + rotation @ offset
    result = local @ rotation.T + origin, origin
    return result


# 功能：
#   对实际使用的源字节生成摘要，供夹具证据绑定，不证明来源具有飞行资质。
# 输入：
#   value：需要绑定的原始字节。
# 输出：
#   digest：SHA-256 十六进制摘要。
def bytes_digest(value: bytes) -> str:
    digest = hashlib.sha256(value).hexdigest()
    return digest
