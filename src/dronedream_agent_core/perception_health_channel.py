"""Run-bound perception readiness transport; no sensor-age or arming authority."""

import threading
from pathlib import Path

from dronedream_plugin_sdk.protocol import copy_json

from .local_packet_channel import LatestPacketPublisher, LatestPacketReceiver, PacketContract
from .sensor_diagnostics import sensor_issue_codes

HEALTH_CONTRACT = PacketContract(
    "dronedream-perception-health", "dronedream-health-",
    "dronedream.perception-fusion-health.v1",
)
_READINESS_FIELDS = (
    "schema_version", "updated_at_unix_ms", "latest_sequence", "stream_healthy",
    "identity_accepted", "truth_correction_applied", "pose_source",
    "realtime_features_ready", "localization_covariance_m2", "localization_observed_at_unix_ms",
    "stream_age_seconds",
    "perception_observed_at_unix_ms",
)


class PerceptionHealthReceiver(LatestPacketReceiver):
    """Single-consumer readiness endpoint, distinct from the native telemetry domain."""

    # 功能：
    #   创建当前运行独占的感知就绪接收端，不接收其他业务域的报文。
    # 输入：
    #   descriptor：本次运行尚未存在的端点文件。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor: Path):
        super().__init__(descriptor, contract=HEALTH_CONTRACT)
        self._latest = None
        self._generation = self._consumed = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._failure = None
        self._thread = threading.Thread(target=self._receive, name="perception-health", daemon=True)
        self._thread.start()

    # 功能：
    #   从解锁前到控制结束持续排空认证报文，只保留最新来源；不积压一场飞行前的健康状态。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _receive(self) -> None:
        try:
            while not self._stop.is_set():
                packet = super().read_latest()
                if packet is not None:
                    with self._lock:
                        self._latest = packet
                        self._generation += 1
                self._stop.wait(.005)
        except BaseException as error:
            with self._lock:
                self._failure = type(error).__name__
                self._latest = None

    # 功能：
    #   每个已接收版本仅向就绪检查交付一次；相同来源的重发仍由就绪窗口拒绝重复计帧。
    # 输入：
    #   无。
    # 输出：
    #   packet：新接收的独立 JSON 副本；没有新版本时为 None。
    def read_latest(self) -> dict | None:
        with self._lock:
            if self._failure is not None:
                raise RuntimeError("PERCEPTION_HEALTH_RECEIVER_FAILED:" + self._failure)
            if self._stop.is_set() or self._generation == self._consumed:
                return None
            self._consumed = self._generation
            packet = copy_json(self._latest, limit=4096)
        return packet

    # 功能：
    #   冻结控制结束瞬间已经收到的最新状态；不等待未来好帧、不读旧文件、不更新时间戳。
    # 输入：
    #   无。
    # 输出：
    #   packet：独立健康副本，未收到或接收端已关闭时为 None。
    def latest_snapshot(self) -> dict | None:
        with self._lock:
            if self._failure is not None:
                raise RuntimeError("PERCEPTION_HEALTH_RECEIVER_FAILED:" + self._failure)
            packet = (None if self._stop.is_set() or self._latest is None
                      else copy_json(self._latest, limit=4096))
        return packet

    # 功能：
    #   先终止唯一收包线程再释放端点，超时不谎报回收完成。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        self._stop.set()
        self._thread.join(1.)
        if self._thread.is_alive():
            raise RuntimeError("PERCEPTION_HEALTH_RECEIVER_NOT_DRAINED")
        super().close()


class PerceptionHealthPublisher(LatestPacketPublisher):
    """Send the original readiness fields before asynchronous durable file publication."""

    # 功能：
    #   初始化仅供本次运行使用的感知发布端，首次绑定后不能跨运行重绑。
    # 输入：
    #   descriptor：执行器持有的感知就绪端点文件。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor: Path):
        super().__init__(descriptor, contract=HEALTH_CONTRACT)

    # 功能：
    #   1. 仅发送就绪判断所需的原始字段，避免诊断数组挤占实时通信预算。
    #   2. 缺失字段保持缺失，不补健康标志、不重写时间；最终就绪仍由消费者验证。
    # 输入：
    #   payload：当前感知健康快照。
    # 输出：
    #   sent：完整报文已交给本机通信时为 True，否则为 False。
    def send(self, payload: dict) -> bool:
        if type(payload) is not dict:
            raise ValueError("PERCEPTION_HEALTH_PACKET_INVALID")
        packet = {key: payload[key] for key in _READINESS_FIELDS if key in payload}
        if "issue_codes" in payload:
            packet["issue_codes"] = sensor_issue_codes(payload["issue_codes"])
        sent = super().send(packet)
        return sent
