"""Native telemetry domain over the shared run-scoped bounded transport."""

from pathlib import Path

from .local_packet_channel import (
    MAX_PACKET_BYTES as MAX_PACKET_BYTES,
)
from .local_packet_channel import (
    LatestPacketPublisher,
    LatestPacketReceiver,
    PacketContract,
)
from .local_packet_channel import (
    _address as _address,
)

_NATIVE = PacketContract("dronedream-native-state", "dronedream-state-",
                         "dronedream.px4-identity-telemetry.v1")


class NativeStateReceiver(LatestPacketReceiver):
    """Native PX4 telemetry domain; payload freshness is checked by the sampler."""
    # 功能：
    #   为原生 PX4 状态建立固定业务域的接收端，不接入模拟器真值替代遥测。
    # 输入：
    #   self：当前原生状态接收器。
    #   descriptor：本次运行独占端点文件。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor: Path):
        super().__init__(descriptor, contract=_NATIVE)


class NativeStatePublisher(LatestPacketPublisher):
    """Publish existing native state, never simulator truth or a renewed sensor age."""
    # 功能：
    #   复用有界本机传输发布原生状态，不重写采集时间或跨运行重绑端点。
    # 输入：
    #   self：当前原生状态发布器。
    #   descriptor：当前接收端的运行专属端点文件。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor: Path):
        super().__init__(descriptor, contract=_NATIVE)
