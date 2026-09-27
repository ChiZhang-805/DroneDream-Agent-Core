"""Run-owned geometry compute process; never owns flight control or devices."""

import json
import multiprocessing
import threading
import time

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .live_map_measurement import (
    MAXIMUM_MEASUREMENT_DISPATCH_AGE_NS,
    LiveMapMeasurementSource,
    MapMeasurementPending,
)

_WIRE_LIMIT = 512 * 1024
_CLOCK_FRESHNESS_SECONDS = 0.030


# 功能：邮箱短时交接迟到时等待新的真实源钟；等待不延长旧钟寿命，也不外推源时间。
# 输入：共享锁、源钟和接收时刻；now、sleep 仅用于单调期限及可重复测试。
# 输出：仍在 30 ms 新鲜度内的源纳秒，或有界等待后返回 None。
def _read_fresh_clock(
    clock_lock, clock_ns, clock_received, *, now=time.monotonic, sleep=time.sleep
):
    deadline = now() + _CLOCK_FRESHNESS_SECONDS
    while True:
        with clock_lock:
            stamp, received = clock_ns.value, clock_received.value
        current = now()
        if stamp <= 0:
            return None  # 父进程明确报告源钟缺失，不能拿旧值等待续期。
        age = current - received
        if 0 <= age <= _CLOCK_FRESHNESS_SECONDS:
            return stamp
        if age < 0 or current >= deadline:
            return None
        sleep(min(0.001, deadline - current))


# 功能：独立进程串行持有配准历史，保留原始观测；不订阅真值、不接触任何执行器。
# 输入：私有管道、只读配置及父进程传递的实际源钟邮箱。
# 输出：逐请求结果或有界异常；收到结束命令后释放管道。
def _worker(pipe, configuration, clock_lock, clock_ns, clock_received):
    try:
        source = LiveMapMeasurementSource(**configuration)
        pipe.send_bytes(b"ready")

        # 功能：读取实际源钟快照，不按主机时间外推；输入：无；输出：新鲜源钟或 None。
        def source_now():
            return _read_fresh_clock(clock_lock, clock_ns, clock_received)

        while True:
            raw = pipe.recv_bytes(_WIRE_LIMIT)
            if raw == b"close":
                return
            try:
                record = decode_json(raw, limit=_WIRE_LIMIT)
                result = source.measure(
                    record, source_now_ns=source_now, monotonic_now=time.monotonic
                )
                # 求解器的不可变元组按 JSON 数组传输；边界仍检查类型、有限数值和大小。
                response = {
                    "kind": "result",
                    "value": json.loads(json.dumps(result, allow_nan=False)),
                }
            except MapMeasurementPending as error:
                response = {"kind": "pending", "value": str(error)[:192]}
            except Exception as error:
                # 私有进程只传固定错误码，不把路径或原始传感器内容透出为诊断。
                code = str(error)
                if (
                    not code
                    or len(code) > 192
                    or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for c in code)
                ):
                    code = "ISOLATED_MAP_COMPUTE_FAILED"
                response = {"kind": "error", "value": code}
            pipe.send_bytes(encode_json(response, limit=_WIRE_LIMIT).encode("utf-8"))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        pipe.close()


class IsolatedMapMeasurementSource:
    """Single outstanding request, no queued old frames and no cross-run history."""

    # 功能：以 spawn 启动纯计算进程，避免继承已运行 Gazebo/遥测线程及其锁。
    # 输入：明确的地图索引、坐标身份和噪声配置；无路线或控制端口。
    # 输出：就绪的串行测量入口；启动失败关闭自身资源，不影响父进程设备。
    def __init__(self, **configuration):
        self.clock_domain = configuration["clock_domain"]
        context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._clock_lock = context.Lock()
        self._clock_ns = context.RawValue("q", 0)
        self._clock_received = context.RawValue("d", 0.0)
        self._pipe, child = context.Pipe()
        self._closed = False
        self._stopped = False
        self._process = context.Process(
            target=_worker,
            args=(child, configuration, self._clock_lock, self._clock_ns, self._clock_received),
            name="map-geometry-compute",
            daemon=True,
        )
        try:
            self._process.start()
            child.close()
            if not self._pipe.poll(15.0) or self._pipe.recv_bytes(64) != b"ready":
                raise RuntimeError("ISOLATED_MAP_STARTUP_FAILED")
        except BaseException:
            child.close()
            self.close()
            raise

    # 功能：把当前真实源钟送入独立计算进程，不续发失效钟，不从主机时间推测源时间。
    # 输入：调用方当前源钟读取函数。
    # 输出：无；畸形或超范围时钟直接拒绝。
    def _update_clock(self, source_now_ns):
        stamp = source_now_ns()
        if stamp is not None and (type(stamp) is not int or not 0 < stamp < 2**63):
            raise ValueError("LIVE_MAP_SOURCE_CLOCK_INVALID")
        with self._clock_lock:
            self._clock_ns.value = 0 if stamp is None else stamp
            self._clock_received.value = time.monotonic()

    # 功能：串行计算并在跨进程返回后再次检查真实源年龄；不把计算完成时刻当拍摄时间。
    # 输入：当前原始记录和实际时钟；无队列、无源时间重写。
    # 输出：同一求解器结果；超时或管道损坏终止纯计算进程，原飞行生命周期仍由外层所有者处理。
    def measure(self, record, *, source_now_ns, monotonic_now):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("ISOLATED_MAP_CONCURRENT_REQUEST")
        outstanding = False
        sender = None
        try:
            if self._closed:
                raise RuntimeError("ISOLATED_MAP_CLOSED")
            payload = encode_json(record, limit=_WIRE_LIMIT).encode("utf-8")
            # 先完成可能耗时的序列化，再采源钟；不能把发送准备耗时当作源钟丢失。
            self._update_clock(source_now_ns)
            outstanding = True
            deadline = time.monotonic() + 2.0
            sent, send_errors = threading.Event(), []

            # 功能：隔离可能阻塞的管道写入；输入：已冻结有界消息；输出：完成信号或原异常。
            def send():
                try:
                    self._pipe.send_bytes(payload)
                except BaseException as error:
                    send_errors.append(error)
                finally:
                    sent.set()

            sender = threading.Thread(target=send, name="map-geometry-send", daemon=True)
            sender.start()
            while not sent.wait(0.002):
                if not self._process.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("ISOLATED_MAP_SEND_TIMEOUT")
                self._update_clock(source_now_ns)
            if send_errors:
                raise send_errors[0]
            while not self._pipe.poll(0.002):
                if not self._process.is_alive() or time.monotonic() > deadline:
                    self.close()
                    raise RuntimeError("ISOLATED_MAP_COMPUTE_TIMEOUT")
                self._update_clock(source_now_ns)
            response = decode_json(self._pipe.recv_bytes(_WIRE_LIMIT), limit=_WIRE_LIMIT)
            # 在确认消息协议前保留 outstanding，畸形回复不能留下可复用的失步通道。
            if (
                type(response) is not dict
                or set(response) != {"kind", "value"}
                or response.get("kind") not in ("result", "pending", "error")
            ):
                raise ValueError("ISOLATED_MAP_RESPONSE_INVALID")
            if response["kind"] in ("pending", "error"):
                code = response["value"]
                if (
                    type(code) is not str
                    or not code
                    or len(code) > 192
                    or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for c in code)
                ):
                    raise ValueError("ISOLATED_MAP_RESPONSE_INVALID")
            else:
                result = response["value"]
                if (
                    type(result) is not dict
                    or type(result.get("source_timestamp_ns")) is not int
                    or not 0 < result["source_timestamp_ns"] < 2**63
                    or result["source_timestamp_ns"]
                    != record["native_odometry_snapshot"]["source_alignment"]["image_timestamp_ns"]
                    or result.get("motion_permission_granted") is not False
                    or result.get("covariance_qualified") is not False
                ):
                    raise ValueError("ISOLATED_MAP_RESPONSE_INVALID")
            outstanding = False
            if response["kind"] == "pending":
                raise MapMeasurementPending(response["value"])
            if response["kind"] == "error":
                raise ValueError(response["value"])
            result = response["value"]
            LiveMapMeasurementSource._require_source_age(
                result["source_timestamp_ns"],
                source_now_ns(),
                maximum_ns=MAXIMUM_MEASUREMENT_DISPATCH_AGE_NS,
            )
            return result
        except (EOFError, BrokenPipeError, OSError) as error:
            self.close()
            raise RuntimeError("ISOLATED_MAP_CHANNEL_FAILED") from error
        finally:
            try:
                if outstanding:
                    # 父钟失败也必须排空/终止在途计算，不能把旧回复留给下一帧。
                    self.close()
                if sender is not None:
                    sender.join(1.0)
                    if sender.is_alive():
                        raise RuntimeError("ISOLATED_MAP_SEND_DID_NOT_STOP")
            finally:
                self._lock.release()

    # 功能：结束只负责数学计算的子进程并核对退出，不终止飞控、感知或用户程序。
    # 输入：无；所有者须在在途 measure 排空之后正常调用。
    # 输出：真实退出状态；超时可终止此唯一自有纯计算进程，不留跨运行历史。
    def close(self):
        if not self._stopped:
            self._closed = True  # 立即拒绝新请求；只有实际退出后才声明回收完成。
            if self._process.pid is not None:
                # 不向可能已停止读取的管道写关闭消息：满管道会把故障清理也永久阻塞。
                # 关闭本端使空闲子进程读到 EOF；在途纯计算有半秒自然退出期限。
                self._pipe.close()
                self._process.join(0.5)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(1.0)
                if self._process.is_alive():
                    raise RuntimeError("ISOLATED_MAP_DID_NOT_STOP")
            self._pipe.close()
            self._stopped = True
        return {"closed": self._stopped, "motion_permission_granted": False}
