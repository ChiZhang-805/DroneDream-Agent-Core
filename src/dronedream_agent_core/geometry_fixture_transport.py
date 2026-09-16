"""Isolated command transport for a non-actuated camera calibration jig.

gz-transport13's blocking Python RequestRaw binding holds the GIL, while
subscription callbacks acquire it. Keep requests in a process without Python
subscriptions; never stall image/pose capture behind that binding. This is NOT
a real flight command transport and exposes only the fixed jig's set_pose.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
import sys
import threading
import time

from dronedream_plugin_sdk.protocol import copy_json, decode_json, encode_json

from .geometry_motion_fixture import RIG_NAME
from .simulation_sensor_frames import _rotation


# 功能：
#   只接受夹具位置与姿态，不允许请求携带实体名、额外字段或非法数值。
# 输入：
#   value：待发送的 position_m 与 orientation_wxyz 对象。
# 输出：
#   None：不返回业务数据。
def validate_fixture_command(value):
    if not isinstance(value, dict) or set(value) != {"position_m", "orientation_wxyz"}:
        raise ValueError("FIXTURE_COMMAND_FIELDS_INVALID")
    _rotation(value["orientation_wxyz"])
    pos = value["position_m"]
    if (not isinstance(pos, list) or len(pos) != 3 or any(type(x) not in (int, float)
            or not math.isfinite(x) or abs(x) > 1e5 for x in pos)):
        raise ValueError("FIXTURE_COMMAND_POSITION_INVALID")


# 功能：
#   在独立解释器及唯一仿真分区内发送固定夹具命令，避免原生阻塞调用阻塞采集回调。
# 输入：
#   connection：有界字节管道的子端。
#   partition：已验证的隔离仿真分区。
#   world：已经验证的世界名。
# 输出：
#   None：不返回业务数据。
def _request_worker(connection, partition, world):
    os.environ["GZ_PARTITION"] = partition
    sys.path.append("/usr/lib/python3/dist-packages")
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.server_control_pb2 import ServerControl
    from gz.transport13 import Node

    node = Node()
    connection.send_bytes(b'{"ready": true}')
    try:
        while True:
            data = connection.recv_bytes(2048)
            if data == b"close":
                return
            if data == b"stop-isolated-server":
                success, response = node.request("/server_control", ServerControl(stop=True),
                                                  ServerControl, Boolean, 500)
                connection.send_bytes(json.dumps({"acknowledged": bool(success and response.data)}
                                                  ).encode("utf-8"))
                continue
            wanted = decode_json(data, limit=2048)
            validate_fixture_command(wanted)
            request = Pose(name=RIG_NAME)
            request.position.x, request.position.y, request.position.z = wanted["position_m"]
            (request.orientation.w, request.orientation.x, request.orientation.y,
             request.orientation.z) = wanted["orientation_wxyz"]
            began = time.perf_counter_ns()
            success, response = node.request(f"/world/{world}/set_pose", request,
                                              Pose, Boolean, 500)
            connection.send_bytes(json.dumps({"acknowledged": bool(success and response.data),
                "service_latency_ms": (time.perf_counter_ns()-began)/1e6}).encode("utf-8"))
    except EOFError:
        return
    finally:
        connection.close()


class FixturePoseRequests:
    # 功能：
    #   启动只服务隔离相机夹具的进程，严格验证就绪握手，失败时回收所拥有的管道。
    # 输入：
    #   self：当前通信客户端。
    #   partition：固定前缀加随机标识的分区名。
    #   world：仅含字母、数字与下划线的世界名。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, partition, world):
        if (sys.platform != "linux" or not isinstance(partition, str)
                or re.fullmatch(r"dronedream-geometry-fixture-[a-f0-9]{32}", partition) is None
                or not isinstance(world, str)
                or re.fullmatch(r"[A-Za-z0-9_]{1,128}", world) is None):
            raise ValueError("FIXTURE_COMMAND_ISOLATION_INVALID")
        context = multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._closed = False
        self._close_result = False
        self.connection, child = context.Pipe(duplex=True)
        try:
            self.process = context.Process(target=_request_worker, args=(child, partition, world),
                                           daemon=True, name="geometry-fixture-commands")
            self.process.start()
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()
        try:
            reply = (decode_json(self.connection.recv_bytes(2048), limit=2048)
                     if self.connection.poll(15) else None)
            ready = (isinstance(reply, dict) and set(reply) == {"ready"}
                     and reply["ready"] is True)
        except (EOFError, OSError, ValueError):
            ready = False
        except BaseException:
            self.close()
            raise
        if not ready:
            self.close()
            raise RuntimeError("FIXTURE_COMMAND_WORKER_NOT_READY")

    # 功能：
    #   1. 串行完成一条请求与其唯一回执，严格校验应答字段、类型和耗时。
    #   2. 超时、断管或非法回执后关闭整条连接，迟到回执不能被下一条命令误认。
    # 输入：
    #   self：当前通信客户端。
    #   payload：已校验且不超过 2048 字节的内部请求。
    #   with_latency：姿态回执是否必须包含服务耗时。
    # 输出：
    #   reply：校验通过的当前请求回执。
    def _exchange(self, payload: bytes, *, with_latency: bool):
        with self._lock:
            if self._closed:
                raise RuntimeError("FIXTURE_COMMAND_WORKER_CLOSED")
            try:
                if not self.process.is_alive():
                    raise RuntimeError("FIXTURE_COMMAND_WORKER_EXITED")
                self.connection.send_bytes(payload)
                if not self.connection.poll(1):
                    raise RuntimeError("FIXTURE_COMMAND_WORKER_TIMED_OUT")
                reply = decode_json(self.connection.recv_bytes(2048), limit=2048)
                expected = ({"acknowledged", "service_latency_ms"}
                            if with_latency else {"acknowledged"})
                if (not isinstance(reply, dict) or set(reply) != expected
                        or type(reply["acknowledged"]) is not bool):
                    raise ValueError("FIXTURE_COMMAND_REPLY_INVALID")
                if with_latency:
                    elapsed = reply["service_latency_ms"]
                    if (type(elapsed) not in (int, float)
                            or not math.isfinite(elapsed) or elapsed < 0):
                        raise ValueError("FIXTURE_COMMAND_REPLY_LATENCY_INVALID")
                return reply
            except BaseException as error:
                # 请求已发出但结果不明确时，无法靠简单重试判断回执归属，必须使通道失效。
                self.close()
                if isinstance(error, (EOFError, OSError, ValueError, OverflowError)):
                    raise RuntimeError("FIXTURE_COMMAND_REPLY_INVALID") from error
                raise

    # 功能：
    #   冻结并校验目标姿态后发送，禁止外部列表在验证和序列化之间改变命令。
    # 输入：
    #   self：当前通信客户端。
    #   wanted：仅包含位置和姿态的目标对象。
    # 输出：
    #   reply：服务是否接受及本次服务耗时；接受不等于夹具已实际到达。
    def request(self, wanted):
        snapshot = copy_json(wanted, limit=2048)
        validate_fixture_command(snapshot)
        payload = encode_json(snapshot, limit=2048).encode("utf-8")
        reply = self._exchange(payload, with_latency=True)
        return reply

    # 功能：
    #   串行且可重复地关闭所拥有的通信进程；宽限期后终止仍存活进程并保留失败结论。
    # 输入：
    #   self：当前通信客户端。
    # 输出：
    #   complete：子进程无需强制终止且正常退出时为 True。
    def close(self):
        with self._lock:
            if self._closed:
                complete = self._close_result
                return complete
            self._closed = True
            forced = False
            try:
                if self.process.is_alive():
                    try:
                        self.connection.send_bytes(b"close")
                    except (OSError, EOFError):
                        forced = True
                self.process.join(2)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(2)
                    forced = True
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(2)
            except (OSError, ValueError):
                forced = True
            finally:
                self.connection.close()
            complete = not forced and self.process.exitcode == 0
            self._close_result = complete
            return complete

    # 功能：
    #   在同一串行通道中停止唯一分区内的隔离仿真服务器，异常或已关闭时返回失败。
    # 输入：
    #   self：当前通信客户端。
    # 输出：
    #   acknowledged：当前停止请求收到真实布尔确认时为 True。
    def stop_server(self):
        try:
            reply = self._exchange(b"stop-isolated-server", with_latency=False)
            acknowledged = reply["acknowledged"]
        except RuntimeError:
            acknowledged = False
        return acknowledged
