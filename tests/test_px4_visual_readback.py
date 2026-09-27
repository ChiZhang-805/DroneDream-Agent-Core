"""The simulator daemon readback never substitutes sends or old output for evidence."""
import asyncio
import os
import socket
import struct
from types import SimpleNamespace

import pytest

from dronedream_agent_core.px4_visual_readback import Px4VisualReadback, decode_px4_response


# 功能：检查响应边界及退出状态；输入：截断、失败、混合响应；输出：严格拒绝。
@pytest.mark.parametrize('data', [b'', b'a', b'text', b'text\x00\x01', b'a\x00b\x00\x00', b'x'*32769],
                         ids=['empty', 'short', 'truncated', 'failed', 'embedded-nul', 'oversize'])
def test_invalid_daemon_reply(data):
    with pytest.raises(ValueError):
        decode_px4_response(data)


# 功能：正确响应去掉协议尾缀；输入：固件文本；输出：文本不修改。
def test_valid_daemon_reply():
    assert decode_px4_response(b'timestamp: 42\n\x00\x00') == 'timestamp: 42\n'


# 功能：仿真连接的身份与资源收尾验证；输入：短读、超时、身份改变；输出：固定命令与必定关闭。
@pytest.mark.parametrize('fault', ['none', 'uid', 'pid', 'timeout', 'oversize', 'cancel'])
def test_socket_query_boundaries(monkeypatch, fault):
    async def scenario():
        output = []
        parts = [b'time', b'stamp: 42\n\x00', b'\x00', b'']
        uid = 1 if fault == 'uid' else 1000
        monkeypatch.setattr(os, 'geteuid', lambda: 1000, raising=False)
        monkeypatch.setattr(socket, 'SO_PEERCRED', 17, raising=False)

        class Reader:
            async def read(self, size):
                if fault == 'cancel':
                    raise asyncio.CancelledError()
                if fault == 'timeout':
                    await asyncio.sleep(1)
                return b'x'*size if fault == 'oversize' else parts.pop(0)

        class Writer:
            def get_extra_info(self, name):
                return SimpleNamespace(getsockopt=lambda *a: struct.pack('3i', 42, uid, 1000))
            def write(self, data):
                output.append(data)
            async def drain(self):
                pass
            def close(self):
                output.append('closed')
            async def wait_closed(self):
                output.append('joined')

        async def connect(path, **kwargs):
            assert path == '/tmp/px4-sock-0'
            return Reader(), Writer()
        monkeypatch.setattr(asyncio, 'open_unix_connection', connect, raising=False)
        client = Px4VisualReadback()
        if fault == 'pid':
            client.peer_pid = 13
        if fault == 'none':
            assert await client.read() == 'timestamp: 42\n'
            assert client.peer_pid == 42 and client.verified_queries == 1
        else:
            error = asyncio.CancelledError if fault == 'cancel' else TimeoutError if fault == 'timeout' else ValueError
            with pytest.raises(error):
                await client.read()
            assert client.verified_queries == 0
        assert output[-2:] == ['closed', 'joined']
        assert all(item in (b'listener vehicle_visual_odometry -n 1\x00', 'closed', 'joined') for item in output)
    asyncio.run(scenario())
