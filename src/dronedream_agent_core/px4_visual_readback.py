"""Bounded read-only PX4 POSIX daemon client for simulator visual odometry.

Uses the same socket protocol as the pinned PX4 client (sock_protocol.cpp and
client.cpp), without launching a process for every measurement. This is actual
uORB output, not a MAVLink send acknowledgement or a cached prediction.
"""

import asyncio
import os
import socket
import struct


# 功能：验证原生守护进程的结束标记及退出状态，不接受截断或隐藏的额外响应。
# 输入：有界响应字节；输出：严格 UTF-8 固件回读，失败不替换为上一次成功数据。
def decode_px4_response(data: bytes) -> str:
    if (not isinstance(data, bytes) or not 2 <= len(data) <= 32768
            or data[-2:] != b'\x00\x00' or b'\x00' in data[:-2]):
        raise ValueError('SITL_MAP_VISION_DAEMON_RESPONSE_INVALID')
    return data[:-2].decode('utf-8', errors='strict')


class Px4VisualReadback:
    # 功能：绑定本次仿真的 PX4 实例；只允许固定的视觉里程计只读命令。
    # 输入：非负实例号；输出：无命令执行、无飞行权限的回读器。
    def __init__(self, instance: int = 0):
        if type(instance) is not int or not 0 <= instance <= 255:
            raise ValueError('SITL_MAP_VISION_INSTANCE_INVALID')
        self.path = f'/tmp/px4-sock-{instance}'
        self.peer_pid = None
        self.verified_queries = 0

    # 功能：直接请求当前固件 uORB 回读，去掉每帧启动 CLI 的开销；不延長原有外层期限。
    # 输入：无；输出：真实固件文本。锁定同 UID/PID，超时或取消均关闭连接。
    async def read(self) -> str:
        writer = None
        try:
            async with asyncio.timeout(.15):
                reader, writer = await asyncio.open_unix_connection(self.path, limit=32768)
                peer = writer.get_extra_info('socket')
                pid, uid, _ = struct.unpack('3i', peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if (uid != os.geteuid() or pid <= 0
                        or self.peer_pid is not None and pid != self.peer_pid):
                    raise ValueError('SITL_MAP_VISION_DAEMON_IDENTITY_CHANGED')
                writer.write(b'listener vehicle_visual_odometry -n 1\x00')
                await writer.drain()
                chunks = bytearray()
                while True:
                    chunk = await reader.read(min(4096, 32769 - len(chunks)))
                    if not chunk:
                        break
                    chunks.extend(chunk)
                    if len(chunks) > 32768:
                        raise ValueError('SITL_MAP_VISION_DAEMON_RESPONSE_OVERSIZED')
                result = decode_px4_response(bytes(chunks))
                self.peer_pid = pid
                self.verified_queries += 1
                return result
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
