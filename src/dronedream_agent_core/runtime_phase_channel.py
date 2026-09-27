"""Live executor-stage context. Neither a phase nor its heartbeat authorizes motion."""

import asyncio
import threading
import time
from pathlib import Path

from dronedream_plugin_sdk.protocol import copy_json

from .local_packet_channel import LatestPacketPublisher, LatestPacketReceiver, PacketContract
from .runtime_phase import phase_context

PHASE_CONTRACT = PacketContract("dronedream-runtime-phase", "dronedream-phase-",
                                "dronedream.runtime-phase-heartbeat.v1")
# 阶段是低频状态机上下文，不是观测或动作租约。20 ms 是传输周期，
# 500 ms 是本机阶段通信失联检测上限（不是允许控制盲飞的时间）。
# 独立原生遥测、碰撞检查及短期安全指令仍各自执行更短的原始期限。
MAXIMUM_PHASE_HEARTBEAT_AGE_SECONDS = .5


class RuntimePhaseReceiver(LatestPacketReceiver):
    # 功能：
    #   建立当前回合独占阶段端点，重放、其他回合和其他数据域均由通信合同拒绝。
    # 输入：
    #   descriptor：当前观察器拥有的新端点路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor: Path):
        super().__init__(descriptor, contract=PHASE_CONTRACT)
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._failure = None
        self._thread = threading.Thread(target=self._receive, name="phase-receiver", daemon=True)
        self._thread.start()

    # 功能：
    #   独立排空通信队列并仅保留最新报文，避免主观察循环做几何检查时旧报文塞满队列。
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
                self._stop.wait(.005)
        except BaseException as error:
            with self._lock:
                self._failure = f"{type(error).__name__}:{error}"
                self._latest = None

    # 功能：
    #   返回原发送时刻计龄的阶段；重复读取缓存不刷新来源年龄，不回退读取旧文件。
    # 输入：
    #   path：保持与文件观察器兼容的参数，不用于读取。
    # 输出：
    #   context：阶段标签及来源年龄；坏钟或无包时为未知。
    def read_context(self, path: Path) -> dict:
        with self._lock:
            packet = self._latest
        if self._failure is not None:
            raise RuntimeError("RUNTIME_PHASE_RECEIVER_FAILED:" + self._failure)
        if packet is None:
            return phase_context(None)
        age = time.time() - packet["updated_at_unix_ms"] / 1000.
        if not 0 <= age <= MAXIMUM_PHASE_HEARTBEAT_AGE_SECONDS:
            return phase_context(None)
        context = {**phase_context(packet.get("state")), "source_age_seconds": age}
        return context

    # 功能：
    #   获取仍在线的执行器所广播的原始目标，不读取诊断文件、不续目标或动作期限。
    # 输入：
    #   无。
    # 输出：
    #   target：独立目标对象；没有当前上下文时抛出异常，调用方不能据此产生运动。
    def read_target(self) -> dict:
        with self._lock:
            packet = self._latest
            failed = self._failure
        if self._stop.is_set() or failed is not None:
            raise RuntimeError("RUNTIME_TARGET_CHANNEL_UNAVAILABLE")
        if packet is None or packet.get("control_target") is None:
            raise FileNotFoundError("RUNTIME_TARGET_NOT_YET_PUBLISHED")
        age = time.time() - packet["updated_at_unix_ms"] / 1000.
        if not 0 <= age <= MAXIMUM_PHASE_HEARTBEAT_AGE_SECONDS:
            raise ValueError("RUNTIME_TARGET_CONTEXT_EXPIRED")
        target = copy_json(packet["control_target"], limit=16 * 1024)
        if type(target) is not dict:
            raise ValueError("RUNTIME_TARGET_CONTEXT_INVALID")
        return target

    # 功能：只返回在线执行端已实际接收的模型回执，不把阶段心跳时间当作回执时间。
    # 输入：无；输出：独立小列表；丢包、过期或关闭返回空，执行端仍独立复核接管。
    def read_model_applications(self):
        with self._lock:
            packet, failed = self._latest, self._failure
        if self._stop.is_set() or failed is not None or packet is None:
            return []
        age = time.time() - packet["updated_at_unix_ms"] / 1000.
        if not 0 <= age <= MAXIMUM_PHASE_HEARTBEAT_AGE_SECONDS:
            return []
        rows = copy_json(packet.get("model_applications", []), limit=2048)
        if type(rows) is not list or len(rows) > 3:
            raise ValueError("RUNTIME_MODEL_APPLICATIONS_INVALID")
        return rows

    # 功能：
    #   停止唯一收包线程后关闭端点，未排空时保留所有权并明确报错。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        self._stop.set()
        self._thread.join(1.)
        if self._thread.is_alive():
            raise RuntimeError("RUNTIME_PHASE_RECEIVER_NOT_DRAINED")
        super().close()


class RuntimePhaseBroadcaster:
    # 功能：
    #   为有限观察器建立阶段广播器，传递当前阶段而非声称飞行控制循环仍有进展。
    # 输入：
    #   descriptors：最多四个本回合观察器端点。
    #   snapshots：当前执行器拥有的状态快照。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptors: list[Path], snapshots, *, model_applications=None):
        if model_applications is not None and not callable(model_applications):
            raise ValueError("RUNTIME_MODEL_APPLICATION_READER_INVALID")
        if not 1 <= len(descriptors) <= 4 or len(set(descriptors)) != len(descriptors):
            raise ValueError("RUNTIME_PHASE_ENDPOINTS_INVALID")
        self._publishers = []
        try:
            for path in descriptors:
                self._publishers.append(LatestPacketPublisher(path, contract=PHASE_CONTRACT))
        except BaseException:
            for publisher in self._publishers:
                publisher.close()
            raise
        self._snapshots = snapshots
        self._model_applications = model_applications
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._close_error = None
        self._last_broadcast = None
        self._maximum_gap_seconds = 0.

    # 功能：
    #   启动唯一阶段传输线程，使非实时的连接和日志工作不阻塞阶段标签传递。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def start(self) -> None:
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("RUNTIME_PHASE_BROADCASTER_ALREADY_USED")
        self._thread = threading.Thread(target=self._run, name="executor-phase", daemon=True)
        self._thread.start()

    # 功能：
    #   每二十毫秒传递内存中的当前阶段；它不是控制进展，不能续传感器或执行指令的租期。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _run(self) -> None:
        try:
            while True:
                closing = self._stop.is_set()
                state = self._snapshots.phase()
                if state:
                    now = time.monotonic()
                    if self._last_broadcast is not None:
                        self._maximum_gap_seconds = max(
                            self._maximum_gap_seconds, now - self._last_broadcast)
                    self._last_broadcast = now
                    payload = {"schema_version": PHASE_CONTRACT.payload_schema,
                               "updated_at_unix_ms": int(time.time() * 1000), "state": state,
                               "control_target": self._snapshots.control_target()}
                    if self._model_applications is not None:
                        payload["model_applications"] = [
                            list(row) for row in self._model_applications()]
                    for publisher in self._publishers:
                        publisher.send(payload)
                # 关闭时同一写线程发送最终阶段，避免跨线程并发写同一个发布套接字。
                if closing:
                    break
                self._stop.wait(.02)
        except BaseException as error:
            self._close_error = error

    # 功能：
    #   停止任务并关闭所有发布端；传播发送失败供最终验收记录。
    # 输入：
    #   无。
    # 输出：
    #   summary：广播已真实结束时的收尾回执。
    async def close(self) -> dict:
        if self._closed:
            if self._close_error is not None:
                raise self._close_error
            return self.summary()
        self._stop.set()
        try:
            if self._thread is not None:
                await asyncio.to_thread(self._thread.join, 2.)
                if self._thread.is_alive():
                    raise RuntimeError("RUNTIME_PHASE_BROADCASTER_NOT_DRAINED")
            if self._close_error is not None:
                raise self._close_error
        except BaseException as error:
            self._close_error = error
            raise
        finally:
            # 超时时保留运行线程与套接字的所有权，不能一边发送一边关闭。
            if self._thread is None or not self._thread.is_alive():
                for publisher in self._publishers:
                    publisher.close()
                self._closed = True
        summary = self.summary()
        return summary

    # 功能：
    #   保存实际发送、拥塞丢包及最大间隔，发送成功不等于控制成功。
    # 输入：
    #   无。
    # 输出：
    #   summary：广播资源状态和各端点计数。
    def summary(self) -> dict:
        summary = {"complete": self._closed and self._close_error is None,
                   "transport": "run-scoped-datagram",
                   "maximum_broadcast_gap_seconds": self._maximum_gap_seconds,
                   "endpoints": [{"sent": p.sent, "dropped": p.dropped}
                                 for p in self._publishers]}
        return summary
