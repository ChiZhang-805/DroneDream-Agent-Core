"""Bounded same-process render preparation, before spawning a flight vehicle.

Only a newly created, collision-free diagnostic rig is moved. The flight model,
installed assets and sensor inputs are never touched. A completed sweep is not
proof that every shader is warm, nor image equivalence or flight qualification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

from .gazebo_subscriptions import GazeboSubscriptions, subscription_shutdown_is_complete
from .plugin_files import check_plain_plugin_path, read_plugin_file
from .render_camera_sweep import swept_pose
from .simulation_camera_profile import camera_configuration
from .xml_values import parse_xml


# 功能：
#   以范围比较检查有限实数，避免超大整数转浮点时溢出，同时排除布尔伪数值。
# 输入：
#   value：待检查的时间或位姿分量。
# 输出：
#   valid：数值能由有限双精度浮点表示时为 True。
def _finite_real(value) -> bool:
    valid = type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max
    return valid


# 功能：
#   在读取现成结果前验证时间水位和等待预算，生成单调时钟截止时间。
# 输入：
#   after：要求结果接收时间严格晚于的水位，可用负值表示起始前。
#   timeout：大于零且不超过 60 秒的等待预算。
# 输出：
#   deadline：当前单调时钟加等待预算后的截止秒数。
def _wait_deadline(after, timeout) -> float:
    if (not _finite_real(after) or type(timeout) not in (int, float)
            or not 0 < timeout <= 60):
        raise ValueError("RENDER_WARMUP_WAIT_BOUNDARY_INVALID")
    deadline = time.monotonic() + timeout
    return deadline


# 功能：
#   1. 核对真实来源摘要和相机配置，只复制传感器及无歧义位姿到静态诊断实体。
#   2. 使用独立实体和主题，不复制飞控、碰撞或可见几何；建好实体不代表完成预热。
# 输入：
#   source：最多 1 MiB 的来源相机 SDF 字节。
#   expected_sha256：调用方明确绑定的来源摘要。
#   identity：32 位小写十六进制诊断身份。
#   include_depth：是否准备深度流，启用后实体需保留至仿真退出。
# 输出：
#   prepared：诊断 SDF 字节及不授予飞行资格的构造回执。
def prepare_rig(source: bytes, expected_sha256: str, identity: str,
                *, include_depth: bool = False) -> tuple[bytes, dict]:
    if type(include_depth) is not bool:
        raise ValueError("RENDER_WARMUP_DEPTH_CHOICE_INVALID")
    if (type(source) is not bytes or len(source) > 1024 * 1024
            or not isinstance(identity, str) or re.fullmatch(r"[a-f0-9]{32}", identity) is None
            or not isinstance(expected_sha256, str)
            or hashlib.sha256(source).hexdigest() != expected_sha256):
        raise ValueError("RENDER_WARMUP_SOURCE_OR_IDENTITY_INVALID")
    config = camera_configuration(source)
    source_root = parse_xml(source, maximum_bytes=1024 * 1024)
    model = source_root.find("model")
    links = model.findall("link")
    if (len(links) != 1 or model.findall(".//plugin") or model.findall(".//frame")
            or model.findall(".//model")):
        raise ValueError("RENDER_WARMUP_REQUIRES_EXPLICIT_CAMERA_LINK")
    # 相对命名坐标需要完整 SDF 求解；重复位姿也不能任选一个当作真实标定。
    if any(len(parent.findall("pose")) > 1 for parent in model.iter()):
        raise ValueError("RENDER_WARMUP_AMBIGUOUS_CAMERA_POSE")
    for pose in model.findall(".//pose"):
        values = (pose.text or "").split()
        if pose.attrib or len(values) != 6 or any(not math.isfinite(float(v)) for v in values):
            raise ValueError("RENDER_WARMUP_UNSUPPORTED_CAMERA_POSE")
    if (model.find("pose") is not None
            and any(float(v) != 0 for v in model.findtext("pose").split())):
        raise ValueError("RENDER_WARMUP_UNSUPPORTED_CAMERA_MODEL_OFFSET")
    root = ET.Element("sdf", version=source_root.get("version", "1.9"))
    name = "dronedream_render_warmup_" + identity
    rig = ET.SubElement(root, "model", name=name)
    ET.SubElement(rig, "static").text = "true"
    link = ET.SubElement(rig, "link", name="camera_link")
    if links[0].find("pose") is not None:
        link.append(copy.deepcopy(links[0].find("pose")))
    streams = {}
    for original in links[0].findall("sensor"):
        # 已准备的深度资源销毁再创建曾触发当前后端退出；显式深度预热保留实体到世界退出。
        if original.get("type") != "camera" and not include_depth:
            continue
        sensor = copy.deepcopy(original)
        kind = "rgb" if original.get("type") == "camera" else "depth"
        # 诊断身份必须独立；仅更换名字不能替代深度资源生命周期管理。
        diagnostic_name = "dronedream_warmup_" + identity + "_" + kind
        sensor.set("name", diagnostic_name)
        topic = "/dronedream/render-warmup/" + identity + "/" + kind
        old_topics = sensor.findall("topic")
        if len(old_topics) > 1:
            raise ValueError("RENDER_WARMUP_AMBIGUOUS_SENSOR_TOPIC")
        if old_topics:
            sensor.remove(old_topics[0])
        ET.SubElement(sensor, "topic").text = topic
        link.append(sensor)
        fields = config[original.get("name")]
        streams[kind] = {"topic": topic, "width": fields["width"], "height": fields["height"],
                        "source_sensor_name": original.get("name"),
                        "diagnostic_sensor_name": diagnostic_name}
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    prepared = content, {"entity_name": name, "streams": streams,
        "source_camera_sha256": expected_sha256, "rig_sha256": hashlib.sha256(content).hexdigest(),
        "source_sensors_configuration": config,
        "prepared_streams": [kind for kind in ("rgb", "depth") if kind in streams],
        "depth_pipeline_prepared": False, "depth_preparation_requested": include_depth,
        "collision_free": True, "control_plugins": False,
        "diagnostic_only": True, "flight_qualification_granted": False}
    return prepared


# 功能：
#   为固定位置的诊断相机生成三个俯仰层、每层八个偏航视角，不生成无人机航迹。
# 输入：
#   position：世界坐标中的 x、y、z 米制位置。
# 输出：
#   views：24 个位置不变的 xyz、roll、pitch、yaw 六元组。
def warmup_views(position: tuple[float, float, float]) -> tuple[tuple, ...]:
    if (not isinstance(position, (list, tuple)) or len(position) != 3
            or any(not _finite_real(v) for v in position)):
        raise ValueError("RENDER_WARMUP_POSITION_INVALID")
    # 只旋转诊断相机，保留所在位置和原世界几何；不触碰飞行实体。
    views = tuple((*position, 0., pitch, math.radians(yaw))
        for pitch in (0., math.pi / 6, -math.pi / 6) for yaw in range(0, 360, 45))
    return views


# 功能：
#   核对来源、准备模式、实体终态及各订阅实际排空，再允许完成预热门禁。
#   不凭 complete 单一标记放行，也不授予飞机或模型资格。
# 输入：
#   receipt：当前预热运行的回执对象。
#   source_sha256：本次明确选择的相机来源摘要。
#   include_depth：本次要求的深度准备模式。
# 输出：
#   ready：来源、模式及退出证据一致时为 True。
def warmup_receipt_ready(receipt: object, *, source_sha256: str, include_depth: bool) -> bool:
    if (not isinstance(receipt, dict) or type(include_depth) is not bool
            or not isinstance(source_sha256, str)
            or re.fullmatch(r"[a-f0-9]{64}", source_sha256) is None):
        ready = False
        return ready
    shutdown = receipt.get("native_subscriptions")
    ready = (
        receipt.get("complete") is True
        and receipt.get("failure") is None
        and receipt.get("source_camera_sha256") == source_sha256
        and receipt.get("depth_preparation_requested") is include_depth
        and receipt.get("depth_pipeline_prepared") is include_depth
        and receipt.get("retained_until_simulation_exit") is include_depth
        and receipt.get("removed_and_absence_observed") is (not include_depth)
        and receipt.get("prepared_streams") == (["rgb", "depth"] if include_depth else ["rgb"])
        and subscription_shutdown_is_complete(shutdown)
        and shutdown["subscribed_count"] == (3 if include_depth else 2)
        and receipt.get("flight_vehicle_absent") is True
        and receipt.get("no_pixels_sent_to_policy") is True
        and receipt.get("flight_qualification_granted") is False
    )
    return ready


class WarmupFrames:
    """Small metadata-only buffer; no diagnostic pixels enter model channels."""

    # 功能：
    #   独立保存预期流尺寸并初始化有界元数据历史，不持有诊断图像像素。
    # 输入：
    #   self：待初始化的帧接收器。
    #   streams：RGB、深度流中的一个或两个及其正整数宽高。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, streams: dict):
        if (not isinstance(streams, dict) or not streams or set(streams) - {"rgb", "depth"}
                or any(not isinstance(value, dict)
                       or any(type(value.get(k)) is not int or not 0 < value[k] <= 8192
                              for k in ("width", "height")) for value in streams.values())):
            raise ValueError("RENDER_WARMUP_STREAM_CONFIGURATION_INVALID")
        self.streams = copy.deepcopy(streams)
        self.condition = threading.Condition()
        self.frames = {key: [] for key in streams}
        self.error = None

    # 功能：
    #   验证图像外形与递增来源时钟，只记录接收时间和来源时间；错误锁存并唤醒等待者。
    #   这里只确认流进度，不验证像素正确性或曝光与位姿严格同步。
    # 输入：
    #   self：本次预热帧接收器。
    #   kind：已注册的 RGB 或深度流名称。
    #   message：原生图像消息。
    #   received：消息抵达的单调时钟秒数。
    # 输出：
    #   None：不返回业务数据。
    def observe(self, kind, message, received: float) -> None:
        with self.condition:
            try:
                if self.error is not None:
                    return
                expected = self.streams[kind]
                sec, nsec = message.header.stamp.sec, message.header.stamp.nsec
                history = self.frames[kind]
                if (not _finite_real(received) or received < 0
                        or type(sec) is not int or sec < 0
                        or type(nsec) is not int or not 0 <= nsec < 1_000_000_000
                        or any(type(v) is not int or v <= 0
                               for v in (message.width, message.height, message.step))
                        or (message.width, message.height)
                        != (expected["width"], expected["height"])
                        or len(message.data) != message.step * message.height
                        or len(history) >= 4096):
                    raise ValueError("invalid frame")
                stamp = sec * 1_000_000_000 + nsec
                if history and (stamp <= history[-1][1] or received < history[-1][0]):
                    self.error = "RENDER_WARMUP_STREAM_CLOCK_NOT_ADVANCING"
                else:
                    history.append((received, stamp))
            except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                self.error = "RENDER_WARMUP_STREAM_INVALID_OR_FULL"
            finally:
                self.condition.notify_all()

    # 功能：
    #   在有界等待内确认每条流都有水位之后的两帧；任何锁存错误先于已有计数检查。
    # 输入：
    #   self：本次预热帧接收器。
    #   after：原生位姿回读完成后的接收水位。
    #   timeout：等待新帧的秒数预算。
    # 输出：
    #   counts：各流中接收时间严格晚于水位的独立帧数。
    def require_progress(self, after: float, timeout: float = 5.) -> dict:
        deadline = _wait_deadline(after, timeout)
        with self.condition:
            while True:
                if self.error:
                    raise ValueError(self.error)
                counts = {key: sum(t > after for t, _ in frames)
                          for key, frames in self.frames.items()}
                # 两个来源时刻必须不同；抵达在位姿确认之后不等于硬件曝光严格同步。
                if min(counts.values()) >= 2:
                    return counts
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("RENDER_WARMUP_FRESH_FRAMES_MISSING")
                self.condition.wait(remaining)


class WarmupPose:
    """Live pose/info readback; scene/info only describes creation-time poses."""

    # 功能：
    #   为独立诊断实体建立位姿回读缓存，分配原生 ID 后才接纳实际匹配结果。
    # 输入：
    #   self：待初始化的位姿接收器。
    #   name：非空诊断实体名称。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, name: str):
        if not isinstance(name, str) or re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", name) is None:
            raise ValueError("RENDER_WARMUP_POSE_NAME_INVALID")
        self.name, self.entity_id = name, None
        self.condition = threading.Condition()
        self.latest = None
        self.message_count = 0
        self.named_pose = None
        self.error = None

    # 功能：
    #   读取自有实体的实时位姿，验证有限坐标与单位四元数；畸形消息锁存失败并唤醒等待者。
    #   不使用 scene/info 中创建时的旧位姿，也不将非法数值写入诊断回执。
    # 输入：
    #   self：本次预热位姿接收器。
    #   message：原生 pose/info 消息。
    # 输出：
    #   None：不返回业务数据。
    def observe(self, message) -> None:
        received = time.monotonic()
        with self.condition:
            self.message_count += 1
            try:
                if self.error is not None:
                    return
                if not _finite_real(received) or received < 0:
                    raise ValueError("invalid receive clock")
                selected = [pose for pose in message.pose if pose.name == self.name]
                if len(selected) > 1:
                    raise ValueError("ambiguous named pose")
                if not selected:
                    return
                pose = selected[0]
                xyz = (pose.position.x, pose.position.y, pose.position.z)
                quaternion = (pose.orientation.w, pose.orientation.x,
                              pose.orientation.y, pose.orientation.z)
                if (type(pose.id) is not int or pose.id <= 0
                        or any(not _finite_real(value) for value in (*xyz, *quaternion))
                        or abs(math.hypot(*quaternion)-1) > 1e-8):
                    raise ValueError("invalid named pose")
                self.named_pose = {"id": pose.id, "name": pose.name, "position": xyz}
                if type(self.entity_id) is int and pose.id == self.entity_id:
                    self.latest = received, xyz, quaternion
            except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                self.error = "RENDER_WARMUP_LIVE_POSE_INVALID"
                self.latest = self.named_pose = None
            finally:
                self.condition.notify_all()

    # 功能：
    #   有界等待自有实体的新位姿，按 0.00001 米位置容差及符号等价单位四元数核对。
    # 输入：
    #   self：本次预热位姿接收器。
    #   xyz：需要确认的世界坐标米制位置。
    #   quaternion：需要确认的 wxyz 单位四元数。
    #   after：位姿设置回执之后的接收水位。
    #   timeout：等待实际回读的秒数预算。
    # 输出：
    #   observed：已匹配位置、四元数和实际接收时间。
    def require_match(self, xyz, quaternion, after: float, timeout: float = 3.) -> dict:
        deadline = _wait_deadline(after, timeout)
        if (not isinstance(xyz, (tuple, list)) or len(xyz) != 3
                or not isinstance(quaternion, (tuple, list)) or len(quaternion) != 4
                or any(not _finite_real(v) for v in (*xyz, *quaternion))
                or abs(math.hypot(*quaternion)-1) > 1e-8):
            raise ValueError("RENDER_WARMUP_POSE_TARGET_INVALID")
        xyz, quaternion = tuple(xyz), tuple(quaternion)
        with self.condition:
            while True:
                if self.error is not None:
                    raise ValueError(self.error)
                if self.latest is not None and self.latest[0] > after:
                    received, actual_xyz, actual_quaternion = self.latest
                    finite = all(_finite_real(v) for v in (*actual_xyz, *actual_quaternion))
                    normalized = finite and abs(math.hypot(*actual_quaternion)-1) <= 1e-8
                    distance = (math.hypot(*(a-b for a, b in zip(actual_xyz, xyz, strict=True)))
                                if finite else math.inf)
                    aligned = (abs(sum(a*b for a, b in zip(
                        actual_quaternion, quaternion, strict=True))) if normalized else 0.)
                    if normalized and distance <= 1e-5 and abs(aligned-1) <= 1e-8:
                        observed = {"position": actual_xyz, "orientation_wxyz": actual_quaternion,
                                "pose_receive_monotonic": received}
                        return observed
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("RENDER_WARMUP_LIVE_POSE_NOT_CONFIRMED")
                self.condition.wait(remaining)


# 功能：
#   1. 仅在飞机尚未生成时创建自有诊断相机，用原生位姿回读和后续帧完成 24 个视角准备。
#   2. 失败也退订和保存回执；RGB 实体须删除并观察缺席，深度实体保留至仿真退出。
#   3. 原生 RPC 在控制线程执行，不在图像回调内执行；诊断像素不进入模型通道。
# 输入：
#   gz_binary：当前运行使用的 Gazebo 命令路径。
#   world：原生世界名称。
#   flight_vehicle：必须尚未存在的飞行实体名称。
#   source：明确绑定的来源相机文件。
#   expected_sha256：来源内容的预期摘要。
#   position：诊断相机固定的世界位置。
#   output：本次独占的新预热输出目录。
#   include_depth：是否预热并保留深度资源。
# 输出：
#   receipt：完成视角、流进度、原生位姿和清理结果，不授予飞行资格。
def execute_warmup(*, gz_binary: str, world: str, flight_vehicle: str,
                   source: Path, expected_sha256: str, position: tuple,
                   output: Path, include_depth: bool = False) -> dict:
    if (not isinstance(gz_binary, str) or not gz_binary.strip() or "\x00" in gz_binary
            or not isinstance(world, str) or re.fullmatch(r"[a-zA-Z0-9_]{1,128}", world) is None
            or not isinstance(flight_vehicle, str)
            or re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", flight_vehicle) is None):
        raise ValueError("RENDER_WARMUP_WORLD_OR_VEHICLE_INVALID")
    views = warmup_views(position)
    position = tuple(position)
    content = read_plugin_file(source, limit=1024 * 1024)
    rig, receipt = prepare_rig(content, expected_sha256, uuid.uuid4().hex,
                               include_depth=include_depth)
    check_plain_plugin_path(output)
    from google.protobuf import text_format
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.empty_pb2 import Empty
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.msgs10.entity_pb2 import Entity
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.scene_pb2 import Scene
    from gz.transport13 import Node

    output.mkdir(parents=True, exist_ok=False)
    with (output / "diagnostic-camera.sdf").open("xb") as stream:
        stream.write(rig)
    name = receipt["entity_name"]
    frames = WarmupFrames(receipt["streams"])
    live_pose = WarmupPose(name)
    owner = GazeboSubscriptions(Node())
    deadline = time.monotonic() + 100
    started = time.monotonic()
    vehicle_absence_verified = False
    owned_entity_id = None

    # 功能：
    #   按当前世界发送有执行时限的原生 RPC，解析回复并要求布尔应答明确成功。
    # 输入：
    #   service：本函数固定使用的场景、创建、位姿或删除服务后缀。
    #   request：待发送的 protobuf 请求对象。
    #   response：预期 protobuf 应答类型。
    # 输出：
    #   answer：通过执行状态与服务拒绝检查的原生应答。
    def rpc(service, request, response):
        # 原生同步调用在此执行，不占用图像回调解释器。
        result = subprocess.run([gz_binary, "service", "-s", f"/world/{world}/{service}",
            "--reqtype", request.DESCRIPTOR.full_name, "--reptype", response.DESCRIPTOR.full_name,
            "--timeout", "5000", "--req", text_format.MessageToString(request)],
            capture_output=True, text=True, timeout=8, check=False)
        if result.returncode != 0 or len(result.stdout) > 16 * 1024 * 1024:
            raise RuntimeError("RENDER_WARMUP_RPC_FAILED:" + service)
        answer = text_format.Parse(result.stdout, response())
        if response is Boolean and answer.data is not True:
            raise RuntimeError("RENDER_WARMUP_RPC_REJECTED:" + service)
        return answer

    # 功能：
    #   查询原生场景并确认飞机不存在，锁定首次观察到的诊断 ID，拒绝操作同名替换实体。
    # 输入：
    #   无。
    # 输出：
    #   indexed：当前场景按唯一实体名索引的原生模型。
    def scene():
        nonlocal vehicle_absence_verified, owned_entity_id
        models = rpc("scene/info", Empty(), Scene).model
        if any(model.name == flight_vehicle for model in models):
            vehicle_absence_verified = False
            raise ValueError("RENDER_WARMUP_REQUIRES_FLIGHT_VEHICLE_ABSENT")
        indexed = {model.name: model for model in models}
        if len(indexed) != len(models):
            raise ValueError("RENDER_WARMUP_AMBIGUOUS_SCENE_IDENTITY")
        if name in indexed:
            identity = indexed[name].id
            if type(identity) is not int or identity <= 0:
                raise ValueError("RENDER_WARMUP_INVALID_NATIVE_ENTITY_ID")
            if owned_entity_id is not None and identity != owned_entity_id:
                raise ValueError("RENDER_WARMUP_NATIVE_ENTITY_REPLACED")
            owned_entity_id = identity
        vehicle_absence_verified = True
        return indexed

    spawned = False
    removed = False
    retained = False
    failure = None
    interruption = None
    observations = []
    try:
        if name in scene():
            raise ValueError("RENDER_WARMUP_ENTITY_ALREADY_EXISTS")
        for kind, stream in receipt["streams"].items():
            owner.subscribe(Image, stream["topic"],
                lambda message, key=kind: frames.observe(key, message, time.monotonic()))
        owner.subscribe(Pose_V, f"/world/{world}/pose/info", live_pose.observe)
        factory = EntityFactory(sdf=rig.decode("utf-8"), name=name, allow_renaming=False)
        # 第一帧渲染前已设置初始位置，不把原点拍到的画面算作指定位置的准备。
        factory.pose.position.x, factory.pose.position.y, factory.pose.position.z = position
        factory.pose.orientation.w = 1
        spawned = True  # 应答丢失不代表创建没有发生，清理必须保留此次尝试的归属。
        rpc("create", factory, Boolean)
        for view in views:
            if time.monotonic() >= deadline:
                raise TimeoutError("RENDER_WARMUP_TOTAL_DEADLINE")
            xyz, quat = swept_pose(view, 0.)
            request = Pose(name=name)
            request.position.x, request.position.y, request.position.z = xyz
            request.orientation.w, request.orientation.x = quat[:2]
            request.orientation.y, request.orientation.z = quat[2:]
            # 创建回执不等于实体已进入场景；名称必须唯一且匹配，不能移动任意飞行模型。
            models = scene()
            if name not in models:
                raise ValueError("RENDER_WARMUP_SPAWN_NOT_OBSERVED")
            request.id = owned_entity_id
            with live_pose.condition:
                live_pose.entity_id = request.id
            rpc("set_pose", request, Boolean)
            # 首次初始化可能先编译资源；较长预算仅限起飞前，不延长飞行输入的有效期。
            observed = live_pose.require_match(xyz, quat, time.monotonic(),
                timeout=min(15. if not observations else 3., max(.001, deadline-time.monotonic())))
            # 首次帧允许引擎初始化和缓存装载，后续视角维持五秒，始终受整体截止时间约束。
            counts = frames.require_progress(time.monotonic(), timeout=min(
                30. if not observations else 5., max(.001, deadline-time.monotonic())))
            if time.monotonic() >= deadline:
                raise TimeoutError("RENDER_WARMUP_TOTAL_DEADLINE")
            observations.append({"pose_xyz_rpy": view, "live_pose": observed,
                                 "post_readback_frames": counts})
    except BaseException as error:
        failure = type(error).__name__ + ":" + str(error)
        if not isinstance(error, Exception):
            interruption = error
    finally:
        if spawned and not include_depth:
            try:
                models = scene()
                if name in models:
                    rpc("remove", Entity(id=models[name].id, name=name, type=Entity.MODEL), Boolean)
                cleanup_deadline = time.monotonic() + 5
                while name in scene():
                    if time.monotonic() >= cleanup_deadline:
                        raise TimeoutError("RENDER_WARMUP_REMOVAL_NOT_OBSERVED")
                    time.sleep(.05)
                removed = True
            except BaseException as error:
                failure = (failure or "") + ";cleanup:" + type(error).__name__ + ":" + str(error)
                if not isinstance(error, Exception) and interruption is None:
                    interruption = error
        elif spawned:
            # 深度诊断实体无碰撞、可见几何和飞控；保留渲染资源直到自有 Gazebo 退出。
            try:
                retained = name in scene()
                if not retained:
                    raise RuntimeError("RENDER_WARMUP_RETAINED_RIG_MISSING")
            except BaseException as error:
                failure = (failure or "") + ";retention:" + type(error).__name__ + ":" + str(error)
                if not isinstance(error, Exception) and interruption is None:
                    interruption = error
        try:
            shutdown = owner.close()
        except BaseException as error:
            shutdown = {"complete": False, "errors": ["shutdown:" + type(error).__name__]}
            failure = (failure or "") + ";shutdown:" + type(error).__name__ + ":" + str(error)
            if not isinstance(error, Exception) and interruption is None:
                interruption = error
        if (not subscription_shutdown_is_complete(shutdown)
                or shutdown["subscribed_count"] != len(receipt["streams"]) + 1):
            failure = (failure or "") + ";subscription-shutdown-incomplete"
        with frames.condition:
            # 回调可能在最后一次成功读取后失败；排空完成后再次检查才可形成成功回执。
            if frames.error:
                failure = (failure or "") + ";frames:" + frames.error
            summary = {key: {"count": len(values), "maximum_gap_ms": max(
                ((b[0]-a[0])*1000 for a, b in zip(values, values[1:], strict=False)), default=None)}
                for key, values in frames.frames.items()}
        with live_pose.condition:
            if live_pose.error:
                failure = (failure or "") + ";pose:" + live_pose.error
            pose_diagnostics = {"message_count": live_pose.message_count,
                "expected_entity_id": live_pose.entity_id,
                "latest_named_pose": live_pose.named_pose, "latest_match": live_pose.latest}
        complete = (failure is None and (retained if include_depth else removed)
                    and len(observations) == len(views))
        receipt.update(complete=complete,
            failure=failure, removed_and_absence_observed=removed, native_subscriptions=shutdown,
            retained_until_simulation_exit=retained,
            depth_pipeline_prepared=complete and include_depth,
            rendering_inactivity_verified=False,
            elapsed_ms=(time.monotonic()-started)*1000, views=observations, frames=summary,
            world=world, flight_vehicle_absent=vehicle_absence_verified,
            no_pixels_sent_to_policy=True,
            full_shader_coverage_claimed=False)
        receipt["live_pose_diagnostics"] = pose_diagnostics
        try:
            with (output / "receipt.json").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt, indent=2, allow_nan=False))
        except BaseException as receipt_error:
            # 证据磁盘失败不能覆盖原中断；保留异常链，并且不改写已经存在的回执。
            if interruption is not None:
                raise interruption from receipt_error
            if failure is not None:
                raise RuntimeError("RENDER_WARMUP_FAILED:" + failure
                                   + ";receipt-write-failed") from receipt_error
            raise
    if interruption is not None:
        raise interruption
    if not receipt["complete"]:
        raise RuntimeError("RENDER_WARMUP_FAILED:" + str(failure))
    return receipt


# 功能：
#   解析显式仿真、相机摘要、位置及新证据目录，运行预热后返回命令退出码。
# 输入：
#   argv：可选命令参数，省略时读取进程命令行。
# 输出：
#   exit_code：预热成功时为零，失败通过异常使进程非零退出。
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gz", required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--flight-vehicle", required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--position", type=float, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-depth", action="store_true",
        help="Prepare depth too; retain the unsubscribed diagnostic rig until world shutdown.")
    args = parser.parse_args(argv)
    execute_warmup(gz_binary=args.gz, world=args.world, flight_vehicle=args.flight_vehicle,
        source=args.camera, expected_sha256=args.source_sha256,
        position=tuple(args.position), output=args.output, include_depth=args.include_depth)
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
