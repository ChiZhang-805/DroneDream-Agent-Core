"""Operation-scoped pose streams must never revive old frames or leak callbacks."""

import asyncio

import pytest

from dronedream_agent_core.gazebo_pose_sampling import FreshGazeboPoseStream


class Node:
    # 功能：
    #   保存隔离回调并记录订阅次数。
    # 输入：
    #   kind、topic、callback：模拟原生订阅参数。
    # 输出：
    #   accepted：注册成功标识。
    def subscribe(self, kind, topic, callback):
        self.callback = callback
        self.count = getattr(self, "count", 0) + 1
        return True

    # 功能：
    #   记录退订，不连接真实 Gazebo。
    # 输入：
    #   topic：待关闭主题。
    # 输出：
    #   accepted：关闭成功标识。
    def unsubscribe(self, topic):
        self.closed = True
        return True


# 功能：
#   证明重复采样复用连接，超时旧帧不满足下一请求，空帧不拼接。
# 输入：
#   无。
# 输出：
#   None：全部时序及关闭断言通过。
def test_fresh_pose_stream_reuses_connection_without_reusing_messages():
    async def scenario():
        node = Node()
        stream = FreshGazeboPoseStream(node, object, "/pose", lambda message: message)
        node.callback({"old": 1})
        with pytest.raises(TimeoutError):
            await stream.sample(0.001)
        first = asyncio.create_task(stream.sample(1))
        await asyncio.sleep(0)
        node.callback(None)
        node.callback({"frame": 2})
        assert await first == {"frame": 2}
        # 第二个回调已绑定第一请求，即使尚在循环排队也不能交付给新请求。
        node.callback({"late": 3})
        second = asyncio.create_task(stream.sample(1))
        await asyncio.sleep(0)
        assert not second.done()
        with pytest.raises(RuntimeError, match="lifecycle"):
            await stream.sample(1)
        node.callback({"frame": 4})
        assert await second == {"frame": 4}
        assert node.count == 1
        assert stream.close()["complete"] is True
        assert node.closed
        with pytest.raises(RuntimeError, match="lifecycle"):
            await stream.sample(1)
    asyncio.run(scenario())


# 功能：
#   证明坏帧导致当前请求失败，取消与关闭不让下一帧复活原请求。
# 输入：
#   无。
# 输出：
#   None：解析失败和取消断言通过。
def test_pose_stream_error_and_cancel_are_not_silently_replaced():
    async def scenario():
        node = Node()
        def decode(message):
            raise ValueError("bad pose")
        stream = FreshGazeboPoseStream(node, object, "/pose", decode)
        request = asyncio.create_task(stream.sample(1))
        await asyncio.sleep(0)
        node.callback({})
        with pytest.raises(ValueError, match="bad pose"):
            await request
        request = asyncio.create_task(stream.sample(1))
        await asyncio.sleep(0)
        assert stream.close()["complete"] is True
        with pytest.raises(asyncio.CancelledError):
            await request
        node.callback({})
    asyncio.run(scenario())
