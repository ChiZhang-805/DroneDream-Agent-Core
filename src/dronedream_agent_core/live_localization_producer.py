"""Independent, latest-only geometry publication; no route or motion authority."""

from __future__ import annotations

import threading
import time
from collections import deque
from copy import deepcopy

from .depth_sensor_binding import DepthSensorBinding
from .localization_source_channel import encode_localization_source
from .pipeline_timing import sensor_processing_timing
from .runtime_sensor_contracts import RuntimeSensorRegistry
from .sensor_bridge import RawMetricRangeScan


class LiveLocalizationProducer:
    # 功能：独立持有投影器并复用只读原生状态缓冲，不共享避障线程的可变标定或任务状态。
    # 输入：snapshot 返回有界自有图像/时钟快照；其余参数绑定实际地图、机体、安装和通道。
    # 输出：尚未启动的最新帧发布器；不创建原生遥测订阅、不授予运动许可。
    def __init__(self, *, snapshot, native_buffer, publisher, mount, vehicle_id,
                 map_sha256, clock_domain, maximum_acceleration_mps2,
                 maximum_alignment_variance_m2):
        self._snapshot, self._native, self._publisher = snapshot, native_buffer, publisher
        self._mount, self._map, self._domain = mount, map_sha256, clock_domain
        self._acceleration = maximum_acceleration_mps2
        self._variance = maximum_alignment_variance_m2
        self._registry = RuntimeSensorRegistry()
        self._binding = DepthSensorBinding(self._registry, vehicle_id=vehicle_id)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._issue = None
        self._sequence = 0
        self._last_source_ns = -1
        self._last_sent_at = None
        self._sent = 0
        self._max_gap = 0.
        self._prepared = deque(maxlen=16)
        self._phase_ms = {}
        self._thread = threading.Thread(target=self._run, name="live-localization-source",
                                        daemon=True)

    # 功能：启动唯一所有者线程，关闭后的实例不可重用。
    # 输入：无。
    # 输出：无；重复启动由线程生命周期检查拒绝。
    def start(self):
        if self._stop.is_set():
            raise RuntimeError("LOCALIZATION_PRODUCER_CLOSED")
        self._thread.start()

    # 功能：仅处理最新可配对的原始帧，重复读取、投影和发布均不刷新来源时间。
    # 输入：无；快照由接收端在短锁内复制，计算不占用图像接收锁。
    # 输出：无；接收尚未就绪时等待，结构或来源错误向所属线程传播。
    def _step(self):
        began = time.monotonic()
        items = tuple(self._snapshot())
        self._record_phase("snapshot", began)
        if len(items) > 16:
            raise ValueError("LOCALIZATION_PRODUCER_SNAPSHOT_CAPACITY")
        items = tuple((image, clock) for image, clock in items
                      if clock.source_unix_ns > self._last_source_ns)
        if not items:
            return
        began = time.monotonic()
        selected = self._native.select_source_image_pose(
            tuple((clock.source_unix_ns // 1_000_000, clock.simulation_ns)
                  for _, clock in items), clock_domain=self._domain,
            now_unix_ms=time.time_ns() // 1_000_000, now_monotonic=time.monotonic(),
            maximum_range_m=self._mount.maximum_range_m,
            maximum_acceleration_mps2=self._acceleration,
            maximum_alignment_variance_m2=self._variance)
        self._record_phase("native_selection", began)
        if selected is None:
            return
        received_ms, source_ns, aligned = selected
        image, clock = next((image, clock) for image, clock in reversed(items)
                           if clock.source_unix_ns // 1_000_000 == received_ms
                           and clock.simulation_ns == source_ns)
        self._record_phase("selected_frame_age", clock.received_monotonic_seconds)
        if aligned.conservative_variance_m2 > self._variance:
            return
        began = time.monotonic()
        projection = self._binding.project(image)
        # 主控制线程可以复用同一帧投影，但只能取独立副本；不再次耗费一遍投影和配对。
        with self._lock:
            self._prepared.append((clock.source_unix_ns, projection,
                                   self._registry.contract("oakd-lite-depth")))
        self._record_phase("projection", began)
        began = time.monotonic()
        self._sequence += 1
        scan = RawMetricRangeScan(sensor_id="oakd-lite-depth", sequence=self._sequence,
            observed_at_unix_ms=received_ms,
            observed_at_monotonic_seconds=clock.sample_monotonic_seconds,
            body_position_world_enu_m=aligned.pose.position_world_enu_m,
            body_orientation_world_from_body=aligned.pose.orientation_world_from_body,
            body_velocity_world_enu_mps=aligned.pose.velocity_world_enu_mps,
            localization_covariance_m2=aligned.conservative_variance_m2,
            samples=list(projection.samples), source_coverage=projection.source_coverage)
        record = {"schema_version": "dronedream.live-geometry-observation.v1",
            "motion_permission_granted": False, "truth_correction_applied": False,
            "map_sha256": self._map, "calibration_sha256": projection.calibration_sha256,
            "native_pose_binding_sha256": aligned.pose.binding_sha256,
            "source_clock": sensor_processing_timing(clock, processing_monotonic=time.monotonic()),
            "scan": scan.model_dump(mode="json"), "mount": self._mount.model_dump(mode="json"),
            "native_odometry_snapshot": aligned.native_odometry_snapshot}
        self._record_phase("scan_record", began)
        began = time.monotonic()
        payload = encode_localization_source(record)
        self._record_phase("encoding", began)
        if self._stop.is_set():
            return
        began = time.monotonic()
        sent = self._publisher.send(payload)
        self._record_phase("publication", began)
        self._last_source_ns = clock.source_unix_ns
        with self._lock:
            if sent:
                now = time.monotonic()
                if self._last_sent_at is not None:
                    self._max_gap = max(self._max_gap, now - self._last_sent_at)
                self._last_sent_at = now
                self._sent += 1

    # 功能：以固定阶段数累计真实耗时，仅用于定位延迟，不参与权限或来源时间计算。
    # 输入：name：内部固定阶段；began：本阶段真实单调开始时刻。
    # 输出：无；统计只由发布线程写入，关闭并排空后才读取。
    def _record_phase(self, name, began):
        elapsed = (time.monotonic() - began) * 1000
        value = self._phase_ms.setdefault(name, {"count": 0, "total": 0., "maximum": 0.})
        value["count"] += 1
        value["total"] += elapsed
        value["maximum"] = max(value["maximum"], elapsed)

    # 功能：按精确来源纳秒复用已完成的投影及标定，不用最新帧替代指定帧，不共享可变状态。
    # 输入：source_unix_ns：接收账本中指定帧的原始时间。
    # 输出：该帧投影与标定契约的独立副本；不存在则 None，调用方仍须重新核验姿态和新鲜度。
    def prepared(self, source_unix_ns):
        with self._lock:
            item = next((item for item in reversed(self._prepared)
                         if item[0] == source_unix_ns), None)
        return deepcopy(item[1:]) if item is not None else None

    # 功能：独立轮询最新输入，不等待较慢的避障组装、模型推理或证据写盘。
    # 输入：无。
    # 输出：无；意外失败保存原始异常并停止，由运行所有者处理终止。
    def _run(self):
        try:
            while not self._stop.is_set():
                self._step()
                self._stop.wait(.005)
        except Exception as error:
            with self._lock:
                self._issue = error

    # 功能：把后台终止传播到运行所有者，避免只在后台静默丢失定位。
    # 输入：无。
    # 输出：无；发生后台异常时抛出附带原始原因的错误。
    def raise_if_failed(self):
        with self._lock:
            issue = self._issue
        if issue is not None:
            raise RuntimeError("LOCALIZATION_PRODUCER_FAILED") from issue

    # 功能：等待发布线程实际结束后返回统计，不关闭借用的原生缓冲或发布通道。
    # 输入：无。
    # 输出：关闭状态与传输间隔；未排空时明确失败，不能伪报结束。
    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("LOCALIZATION_PRODUCER_DID_NOT_STOP")
        with self._lock:
            return {"sent_frames": self._sent, "projected_frames": self._sequence,
                    "maximum_send_gap_seconds": self._max_gap, "thread_alive": False,
                    "phase_ms": deepcopy(self._phase_ms),
                    "issue": str(self._issue) if self._issue is not None else None}
