"""Bounded native geometry capture for offline estimator calibration.

This records measured sensor-frame rays, including misses, before motion is
authorized. It never changes pose, uncertainty, controls, or learning labels.
The capture is a finite sampled clip, not a promise to record every frame.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path

from dronedream_plugin_sdk.protocol import copy_json, encode_json

from .contracts import CalibratedRangeSensorMount, RawMetricRangeScan
from .hashing import sha256_json
from .runtime_evidence import BoundedRuntimeEvidenceWriter, reserve_new_evidence_files

MAXIMUM_RECORD_BYTES = 128 * 1024
MAXIMUM_CAPTURE_RECORDS = 512  # At most 64 MiB of JSONL, plus a small summary.
MAXIMUM_CAPTURE_RAYS = 512


# 功能：
#   检查内容摘要的语法，不把摘要当作签名或真实飞行资格。
# 输入：
#   value：候选 SHA-256 文本。
# 输出：
#   valid：文本符合小写十六进制摘要格式时为 True。
def _digest(value: str) -> bool:
    valid = isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    return valid


# 功能：
#   在后台检查完整记录预算并绑定全部回放字段，拒绝已有摘要或 JSON 类型的隐式转换。
# 输入：
#   record：尚未附加摘要的几何记录。
# 输出：
#   text：包含自身摘要的紧凑 JSON 文本，连同换行不超过单条记录预算。
def encode_geometry_record(record: dict) -> str:
    if type(record) is not dict or "record_sha256" in record:
        raise ValueError("LOCALIZATION_CAPTURE_RECORD_INVALID")
    try:
        encode_json(record, limit=MAXIMUM_RECORD_BYTES - 1)
    except ValueError as error:
        raise ValueError("LOCALIZATION_CAPTURE_RECORD_TOO_LARGE_OR_INVALID") from error
    value = {**record, "record_sha256": sha256_json(record)}
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text.encode("utf-8")) + 1 > MAXIMUM_RECORD_BYTES:
        raise ValueError("LOCALIZATION_CAPTURE_RECORD_TOO_LARGE")
    return text


class GeometryObservationCapture:
    """Single-producer, bounded sensor replay; no controller or truth feedback."""

    # 功能：
    #   校验采样配置及发布器，独占领取新运行的记录文件，并建立有界后台写入队列。
    # 输入：
    #   self：待初始化的几何采集器。
    #   run_dir：本次运行独立拥有的目录。
    #   map_sha256：已接入地图的内容摘要。
    #   summary_publisher：原子发布完整采集回执的回调。
    #   maximum_records：最多接收的记录数。
    #   minimum_interval_seconds：相邻接收记录的最小来源时间间隔。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, run_dir: Path, *, map_sha256: str,
                 summary_publisher: Callable[[Path, object], None],
                 maximum_records: int = MAXIMUM_CAPTURE_RECORDS,
                 minimum_interval_seconds: float = .2) -> None:
        if not callable(summary_publisher):
            raise ValueError("LOCALIZATION_CAPTURE_PUBLISHER_INVALID")
        if not _digest(map_sha256):
            raise ValueError("LOCALIZATION_CAPTURE_MAP_ID_INVALID")
        if type(maximum_records) is not int or not 1 <= maximum_records <= MAXIMUM_CAPTURE_RECORDS:
            raise ValueError("LOCALIZATION_CAPTURE_RECORD_LIMIT_INVALID")
        if (type(minimum_interval_seconds) not in (int, float)
                or not .2 <= minimum_interval_seconds <= 10.):
            raise ValueError("LOCALIZATION_CAPTURE_PERIOD_INVALID")
        self.path = run_dir / "native-geometry-observations.jsonl"
        self.summary_path = run_dir / "native-geometry-observation-summary.json"
        if self.path.exists() or self.summary_path.exists():
            raise FileExistsError("LOCALIZATION_CAPTURE_MUST_USE_NEW_RUN")
        reserve_new_evidence_files(self.path, self.summary_path)
        self._publish = summary_publisher
        self._map_sha256 = map_sha256
        self.maximum_records = maximum_records
        self.minimum_interval_seconds = minimum_interval_seconds
        self._last_time: float | None = None
        self._last_sequence = -1
        self._accepted = self._sampling_skipped = self._quota_skipped = 0
        self._issue: str | None = None
        self._closed = False
        self._writer = BoundedRuntimeEvidenceWriter(
            self.summary_path, summary_publisher=self._publish_summary,
            serializer=encode_geometry_record, maximum_pending_records=4,
        )

    # 功能：
    #   1. 对有界扫描及安装契约重新验证，保存独立快照，保留来源时间和序列。
    #   2. 采样及总量限制只统计跳过；非法输入或入队失败锁定不完整状态，不授予控制权限。
    # 输入：
    #   self：单生产者使用的采集器。
    #   scan：实测传感器坐标射线及机体估计状态。
    #   mount：与扫描传感器身份匹配的安装外参。
    #   calibration_sha256：当前光学校准摘要。
    #   native_pose_binding_sha256：当前原生位姿绑定摘要。
    #   source_clock：有界来源时钟诊断；None 表示未取得该诊断。
    #   native_odometry_snapshot：原生协方差的独立时钟快照；不宣称与图像同步。
    # 输出：
    #   accepted：本条记录成功进入后台队列时为 True，不代表已经写盘。
    def record(self, scan: RawMetricRangeScan, *, mount: CalibratedRangeSensorMount,
               calibration_sha256: str, native_pose_binding_sha256: str,
               source_clock: dict | None, native_odometry_snapshot: dict | None = None) -> bool:
        if self._closed or self._issue is not None or self._writer.issue is not None:
            return False
        if self._accepted >= self.maximum_records:
            self._quota_skipped += 1
            return False
        # model_copy 和嵌套容器修改能绕过赋值检查；先限制规模，再重新验证独立模型。
        try:
            if (not isinstance(scan, RawMetricRangeScan)
                    or not isinstance(mount, CalibratedRangeSensorMount)
                    or not isinstance(scan.samples, (list, tuple))
                    or len(scan.samples) > MAXIMUM_CAPTURE_RAYS or scan.dynamic_obstacles):
                raise ValueError("capture ray budget exceeded")
            scan = RawMetricRangeScan.model_validate(scan.model_dump(mode="python"), strict=True)
            mount = CalibratedRangeSensorMount.model_validate(
                mount.model_dump(mode="python"), strict=True)
            if source_clock is not None and type(source_clock) is not dict:
                raise ValueError("capture source clock is not an object")
            source_clock = copy_json(source_clock, limit=8192)
            if native_odometry_snapshot is not None:
                if (type(native_odometry_snapshot) is not dict
                        or native_odometry_snapshot.get("image_synchronized") is not False):
                    raise ValueError("capture odometry cannot claim image synchronization")
                native_odometry_snapshot = copy_json(native_odometry_snapshot, limit=8192)
        except (ValueError, TypeError, AttributeError, OverflowError):
            self._issue = "LOCALIZATION_CAPTURE_INPUT_INVALID"
            return False
        source_time = scan.observed_at_monotonic_seconds
        if (not math.isfinite(source_time) or source_time < 0
                or (self._last_time is not None and source_time < self._last_time)
                or scan.sequence < self._last_sequence):
            self._issue = "LOCALIZATION_CAPTURE_SOURCE_REGRESSED"
            return False
        if (scan.sequence == self._last_sequence
                or (self._last_time is not None
                    and source_time - self._last_time < self.minimum_interval_seconds)):
            self._sampling_skipped += 1
            return False
        if (not _digest(calibration_sha256) or not _digest(native_pose_binding_sha256)
                or scan.sensor_id != mount.sensor_id or len(scan.samples) > MAXIMUM_CAPTURE_RAYS
                or scan.dynamic_obstacles):
            self._issue = "LOCALIZATION_CAPTURE_BINDING_OR_RAY_BUDGET_INVALID"
            return False
        # 这里只保存测量及估计值；仿真动态真值由另一条独立流记录，不能反灌控制器。
        record = {
            "schema_version": "dronedream.native-geometry-observation.v1",
            "intended_use": "offline-localization-calibration",
            "motion_permission_granted": False,
            "model_control_qualification_granted": False,
            "truth_correction_applied": False,
            "map_sha256": self._map_sha256,
            "calibration_sha256": calibration_sha256,
            "native_pose_binding_sha256": native_pose_binding_sha256,
            "source_clock": source_clock,
            "scan": scan.model_dump(mode="json"),
            "mount": mount.model_dump(mode="json"),
        }
        if native_odometry_snapshot is not None:
            record["native_odometry_snapshot"] = native_odometry_snapshot
        if not self._writer.submit(self.path, record):
            self._issue = "LOCALIZATION_CAPTURE_ENQUEUE_FAILED"
            return False
        self._accepted += 1
        self._last_time, self._last_sequence = source_time, scan.sequence
        accepted = True
        return accepted

    # 功能：
    #   将队列写入结果与采集状态合并，防止底层成功掩盖入队前的采集失败。
    # 输入：
    #   self：当前采集器。
    #   writer_summary：后台队列的独立状态快照。
    # 输出：
    #   summary：包含配额、采样、错误和离线用途的完整回执。
    def _summary(self, writer_summary: dict) -> dict:
        summary = {**writer_summary,
            "complete": writer_summary["complete"] and self._issue is None,
            "capture_issue": self._issue,
            "intended_use": "offline-localization-calibration",
            "model_control_qualification_granted": False,
            "minimum_interval_seconds": self.minimum_interval_seconds,
            "maximum_records": self.maximum_records,
            "maximum_jsonl_bytes": self.maximum_records * MAXIMUM_RECORD_BYTES,
            "accepted_count": self._accepted,
            "sampling_skipped_count": self._sampling_skipped,
            "quota_skipped_count": self._quota_skipped,
            "quota_reached": self._accepted >= self.maximum_records}
        return summary

    # 功能：
    #   每次发布前先合并采集失败状态，禁止短暂写出未经完整检查的成功回执。
    # 输入：
    #   self：当前采集器。
    #   path：队列固定的回执路径。
    #   writer_summary：待发布的队列状态。
    # 输出：
    #   None：不返回业务数据。
    def _publish_summary(self, path: Path, writer_summary: dict) -> None:
        self._publish(path, self._summary(writer_summary))

    # 功能：
    #   停止接收并限时排空队列，返回真实完成或失败状态，不将离线记录当作飞行证明。
    # 输入：
    #   self：当前采集器。
    # 输出：
    #   summary：发布失败、关闭超时和采集错误均不能被隐藏的最终回执。
    def close(self) -> dict:
        self._closed = True
        summary = self._summary(self._writer.close(timeout_seconds=4.))
        return summary
