"""Move only verified static diagnostic cameras, never a flight-controlled model.

Changing the view exercises lazy GPU work that a fixed-camera benchmark misses.
The resulting measurements are simulator diagnostics, not flight qualification.
"""

from __future__ import annotations

import math
import multiprocessing
import re
import threading
import time
from xml.etree import ElementTree as ET

from dronedream_plugin_sdk.protocol import decode_json, encode_json


# 功能：
#   核对固定诊断相机的身份、静态属性和传感器，再补充姿态服务，不改变地图几何。
# 输入：
#   content：待验证的渲染世界 SDF 字节，最多 32 MiB。
#   count：待移动的诊断相机数量，范围一至八。
# 输出：
#   result：派生世界字节、世界名及已经验证的相机初始姿态列表。
def prepare_camera_sweep(content: bytes, count: int) -> tuple[bytes, str, list[tuple]]:
    if not isinstance(content, bytes) or len(content) > 32 * 1024 * 1024:
        raise ValueError("RENDER_SWEEP_WORLD_SIZE")
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("RENDER_SWEEP_CAMERA_COUNT")
    root = ET.fromstring(content)
    worlds = root.findall("world")
    if len(worlds) != 1:
        raise ValueError("RENDER_SWEEP_REQUIRES_ONE_WORLD")
    world = worlds[0]
    name = world.get("name", "")
    if re.fullmatch(r"[a-zA-Z0-9_]{1,128}", name) is None:
        raise ValueError("RENDER_SWEEP_WORLD_NAME")
    cameras = []
    for index in range(count):
        camera_name = f"render_probe_{index}"
        matches = [m for m in world.findall("model") if m.get("name") == camera_name]
        if len(matches) != 1:
            raise ValueError("RENDER_SWEEP_CAMERA_IDENTITY")
        model = matches[0]
        sensors = model.findall("link/sensor")
        if (model.findtext("static") != "true" or model.findall(".//plugin")
                or model.findall(".//include") or model.findall(".//collision")
                or len(sensors) != 1 or sensors[0].get("type") != "rgbd_camera"
                or sensors[0].findtext("topic") != f"/dronedream/render-probe/{index}"):
            raise ValueError("RENDER_SWEEP_REQUIRES_STATIC_DIAGNOSTIC_CAMERA")
        pose = model.find("pose")
        if pose is None or pose.attrib:
            raise ValueError("RENDER_SWEEP_REQUIRES_WORLD_RPY_POSE")
        coordinates = tuple(float(v) for v in (pose.text or "").split())
        if len(coordinates) != 6 or any(not math.isfinite(v) for v in coordinates):
            raise ValueError("RENDER_SWEEP_REQUIRES_FINITE_POSE")
        cameras.append((camera_name, coordinates))
    if not any(p.get("name") == "gz::sim::systems::UserCommands" for p in world.findall("plugin")):
        ET.SubElement(world, "plugin", name="gz::sim::systems::UserCommands",
                      filename="gz-sim-user-commands-system")
    result = ET.tostring(root, encoding="utf-8", xml_declaration=True), name, cameras
    return result


# 功能：
#   围绕世界竖直轴增加诊断相机偏航，保持原始位置、滚转与俯仰。
# 输入：
#   coordinates：原始 x、y、z 米制坐标及 roll、pitch、yaw 弧度角。
#   delta_yaw：新增偏航角，单位弧度。
# 输出：
#   result：原始位置三元组与合成后的 wxyz 四元数。
def swept_pose(coordinates: tuple, delta_yaw: float) -> tuple[tuple, tuple]:
    if len(coordinates) != 6 or any(type(v) not in (int, float) or not math.isfinite(v)
                                  for v in (*coordinates, delta_yaw)):
        raise ValueError("RENDER_SWEEP_REQUIRES_FINITE_POSE")
    x, y, z, roll, pitch, yaw = coordinates
    if not math.isfinite(yaw + delta_yaw):
        raise ValueError("RENDER_SWEEP_REQUIRES_FINITE_POSE")
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos((yaw + delta_yaw) / 2), math.sin((yaw + delta_yaw) / 2)
    # Gazebo 姿态将局部向量转到世界坐标，不反转四元数方向。
    result = (x, y, z), (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
    return result


# 功能：
#   在独立进程中串行移动已验证的诊断相机，原生阻塞请求不占用图像回调解释器。
# 输入：
#   connection：有界 JSON 字节管道。
#   world：已经验证的世界名。
#   cameras：已冻结的诊断相机名及初始姿态。
# 输出：
#   None：不返回业务数据。
def _request_worker(connection, world: str, cameras: list[tuple]):
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node

    node = Node()
    try:
        while True:
            delta = decode_json(connection.recv_bytes(2048), limit=2048)
            if delta is None:
                return
            request = Pose_V()
            for name, coordinates in cameras:
                position, quaternion = swept_pose(coordinates, delta)
                pose = request.pose.add()
                pose.name = name
                pose.position.x, pose.position.y, pose.position.z = position
                (pose.orientation.w, pose.orientation.x,
                 pose.orientation.y, pose.orientation.z) = quaternion
            started = time.monotonic()
            accepted, reply = node.request(f"/world/{world}/set_pose_vector",
                                          request, Pose_V, Boolean, 2000)
            result = {"transport_accepted": accepted,
                "reply": reply.data if reply is not None else None,
                "request_elapsed_ms": (time.monotonic() - started) * 1000}
            connection.send_bytes(encode_json(result, limit=2048).encode("utf-8"))
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


class CameraSweepClient:
    # 功能：
    #   重新验证世界后创建专属扫描进程，构造或启动失败时关闭两端管道。
    # 输入：
    #   self：当前扫描客户端。
    #   content：已准备世界的 SDF 字节。
    #   count：诊断相机数量。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, content: bytes, count: int):
        # 进程边界再次验证，只给这份文档中的诊断相机开放偏航增量，不接受实体名。
        _, world, cameras = prepare_camera_sweep(content, count)
        context = multiprocessing.get_context("spawn")
        self._parent, child = context.Pipe()
        self._closed = False
        self._lock = threading.RLock()
        self._summary = {"forced": True, "exitcode": None, "complete": False}
        try:
            self._process = context.Process(target=_request_worker, args=(child, world, cameras))
            self._process.start()
        except BaseException:
            self._parent.close()
            raise
        finally:
            child.close()

    # 功能：
    #   串行发送偏航增量并验证当前回执；超时及非法应答后使通道失效，禁止误用迟到结果。
    # 输入：
    #   self：当前扫描客户端。
    #   delta_yaw：相对初始姿态的偏航增量，单位弧度。
    # 输出：
    #   result：运输接受状态、服务应答及请求耗时；不代表相机已经实际转动。
    def request(self, delta_yaw: float) -> dict:
        if (isinstance(delta_yaw, bool) or not isinstance(delta_yaw, int | float)
                or not math.isfinite(delta_yaw)):
            raise ValueError("RENDER_SWEEP_INVALID_YAW")
        payload = encode_json(delta_yaw, limit=2048).encode("utf-8")
        with self._lock:
            if self._closed:
                raise RuntimeError("RENDER_SWEEP_CLIENT_NOT_RUNNING")
            try:
                if not self._process.is_alive():
                    raise RuntimeError("RENDER_SWEEP_CLIENT_NOT_RUNNING")
                self._parent.send_bytes(payload)
                if not self._parent.poll(3):
                    raise TimeoutError("RENDER_SWEEP_RPC_RESULT_TIMEOUT")
                result = decode_json(self._parent.recv_bytes(2048), limit=2048)
                if (not isinstance(result, dict) or set(result) != {
                        "transport_accepted", "reply", "request_elapsed_ms"}
                        or type(result["transport_accepted"]) is not bool
                        or (type(result["reply"]) is not bool and result["reply"] is not None)
                        or (result["transport_accepted"] and result["reply"] is None)):
                    raise ValueError("RENDER_SWEEP_REPLY_INVALID")
                elapsed = result["request_elapsed_ms"]
                if (type(elapsed) not in (int, float)
                        or not math.isfinite(elapsed) or elapsed < 0):
                    raise ValueError("RENDER_SWEEP_REPLY_INVALID")
                return result
            except BaseException as error:
                self.close()
                if isinstance(error, TimeoutError):
                    raise
                if isinstance(error, (EOFError, OSError, ValueError, OverflowError)):
                    raise RuntimeError("RENDER_SWEEP_REPLY_INVALID") from error
                raise

    # 功能：
    #   以有界等待关闭自己的扫描进程，必要时终止该进程，重复调用返回同一清理结论。
    # 输入：
    #   self：当前扫描客户端。
    # 输出：
    #   summary：独立的清理摘要副本，包含强制退出标记、退出码与正常完成标记。
    def close(self) -> dict:
        with self._lock:
            if self._closed:
                summary = dict(self._summary)
                return summary
            self._closed = True
            forced = False
            try:
                if self._process.is_alive():
                    try:
                        self._parent.send_bytes(b"null")
                    except (OSError, EOFError):
                        forced = True
                self._process.join(timeout=3)
                if self._process.is_alive():
                    # 仅回收当前实例拥有的 RPC 客户端，不杀仿真器或其他飞控进程。
                    self._process.terminate()
                    self._process.join(timeout=2)
                    forced = True
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=2)
            except (OSError, ValueError):
                forced = True
            finally:
                self._parent.close()
            self._summary = {"forced": forced, "exitcode": self._process.exitcode,
                             "complete": not forced and self._process.exitcode == 0}
            summary = dict(self._summary)
            return summary
