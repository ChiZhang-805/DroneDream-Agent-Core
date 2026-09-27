"""Isolate Gazebo's GIL-holding payload service calls from flight control."""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
from contextlib import suppress
from pathlib import Path

_LIMIT = 16_384


# 功能：
#   在发送与子进程执行前校验唯一允许的载荷位姿请求，不提供任意服务或代码执行入口。
# 输入：
#   request：世界、物品名称、七个位姿分量、序号及毫秒预算。
# 输出：
#   request：已校验的原请求对象。
def validate_request(request: dict) -> dict:
    if not isinstance(request, dict) or set(request) != {
        "world", "model", "pose", "sequence", "timeout_ms"
    }:
        raise ValueError("invalid payload service request fields")
    for key in ("world", "model"):
        value = request[key]
        if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,512}", value) is None:
            raise ValueError("invalid payload service entity")
    pose = request["pose"]
    if (not isinstance(pose, list) or len(pose) != 7
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in pose)
            or abs(sum(v * v for v in pose[3:]) - 1.0) > 1e-5):
        raise ValueError("invalid payload service pose")
    if type(request["sequence"]) is not int or request["sequence"] < 1:
        raise ValueError("invalid payload service sequence")
    if type(request["timeout_ms"]) is not int or not 1 <= request["timeout_ms"] <= 30_000:
        raise ValueError("invalid payload service timeout")
    return request


class PayloadPoseService:
    # 功能：
    #   创建操作级原生服务拥有者；只有实际请求时启动隔离进程。
    # 输入：
    #   无。
    # 输出：
    #   None：初始化进程、序号和关闭状态。
    def __init__(self):
        self._process = None
        self._sequence = 0
        self._busy = False
        self._closed = False

    # 功能：
    #   串行请求载荷位姿服务；超时、错误或取消后销毁通道，不复用未知执行状态。
    # 输入：
    #   world、model：当前世界与已授权的分离载荷。
    #   pose：位置和四元数七元数组。
    #   timeout_ms：原生请求预算。
    # 输出：
    #   response：与请求序号绑定的服务接纳证据，不能替代实际位姿回读。
    async def set_pose(self, *, world: str, model: str, pose: list, timeout_ms: int) -> dict:
        if self._closed or self._busy:
            raise RuntimeError("payload service is closed or already in use")
        request = validate_request(dict(world=world, model=model, pose=pose,
                                        sequence=self._sequence + 1, timeout_ms=timeout_ms))
        # 子进程启动期间会让出调度；冻结已检查的位姿，不能发送调用方随后改写的坐标。
        request = {**request, "pose": list(request["pose"])}
        self._busy = True
        try:
            if self._process is None:
                # 直接执行本模块文件，避免子进程加载完整 Core 包；不继承控制对象或线程。
                self._process = await asyncio.create_subprocess_exec(
                    sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL, limit=_LIMIT,
                )
                async with asyncio.timeout(5.0):
                    ready = await self._process.stdout.readline()
                if ready not in (b'{"ready":true}\n', b'{"ready":true}\r\n'):
                    raise RuntimeError("payload service worker did not become ready")
            self._sequence += 1
            data = (json.dumps(request, allow_nan=False) + "\n").encode()
            self._process.stdin.write(data)
            async with asyncio.timeout(1.0):
                await self._process.stdin.drain()
            async with asyncio.timeout(timeout_ms / 1000 + 0.5):
                line = await self._process.stdout.readline()
            if not line.endswith(b"\n") or len(line) > _LIMIT:
                raise RuntimeError("payload service response missing or oversized")
            response = json.loads(line)
            if (not isinstance(response, dict) or response.get("sequence") != self._sequence
                    or type(response.get("sequence")) is not int
                    or response.get("accepted") is not True):
                raise RuntimeError("payload service did not confirm the bound request: " +
                                   str(response)[:512])
            return response
        except BaseException:
            await self.close()
            raise
        finally:
            self._busy = False

    # 功能：
    #   结束本对象唯一拥有的服务进程并等到退出，避免取消后遗留请求发送者。
    # 输入：
    #   无。
    # 输出：
    #   None：进程已回收，通道不可重开。
    async def close(self) -> None:
        self._closed = True
        process, self._process = self._process, None
        if process is None:
            return
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        if process.stdin is not None:
            process.stdin.close()
        # 单独等待任务，外层取消不能使拥有者在进程退出前宣称清理完成。
        waiting = asyncio.create_task(process.wait())
        try:
            await asyncio.wait_for(asyncio.shield(waiting), 5.0)
        except asyncio.CancelledError:
            await asyncio.wait_for(asyncio.shield(waiting), 5.0)
            raise


# 功能：
#   只在隔离进程调用 Gazebo 同步位姿服务，控制进程的解释器和遥测不受其 GIL 阻塞。
# 输入：
#   标准输入：有界 JSON 请求行。
# 输出：
#   exit_code：输入结束为零，协议或原生服务错误为一。
def worker_main() -> int:
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.pose_pb2 import Pose
    from gz.transport13 import Node

    node = Node()
    print('{"ready":true}', flush=True)
    sequence = 0
    while True:
        line = sys.stdin.buffer.readline(_LIMIT + 1)
        if not line:
            return 0
        try:
            if len(line) > _LIMIT or not line.endswith(b"\n"):
                raise ValueError("oversized request")
            request = validate_request(json.loads(line))
            if request["sequence"] != sequence + 1:
                raise ValueError("out of order request")
            sequence = request["sequence"]
            message = Pose()
            message.name = request["model"]
            p = request["pose"]
            message.position.x, message.position.y, message.position.z = p[:3]
            (message.orientation.x, message.orientation.y,
             message.orientation.z, message.orientation.w) = p[3:]
            # 专用原生服务同时提交位姿与一次性的零初速度；不退回保留下落速度的旧 set_pose。
            accepted, result = node.request(
                f'/world/{request["world"]}/model/{request["model"]}/place_detached',
                                             message, Pose, Boolean, request["timeout_ms"])
            response = dict(sequence=sequence, accepted=accepted is True and result.data is True,
                            transport_accepted=bool(accepted), native_accepted=bool(result.data))
            print(json.dumps(response, allow_nan=False), flush=True)
            if not response["accepted"]:
                return 1
        except Exception:
            print(json.dumps(dict(sequence=sequence, accepted=False)), flush=True)
            return 1


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("worker entry only")
    raise SystemExit(worker_main())
