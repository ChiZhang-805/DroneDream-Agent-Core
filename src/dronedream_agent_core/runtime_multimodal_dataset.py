"""Bounded, hash-chained recording of synchronized onboard sensor evidence."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from contextlib import suppress
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from pydantic import Field, model_validator

from dronedream_plugin_sdk.protocol import MAX_MESSAGE_BYTES, encode_json

from .contracts import OnboardPerceptionFrame, StrictModel
from .hashing import sha256_json
from .plugin_files import check_plain_plugin_path, read_plugin_file
from .plugin_values import plugin_json_value
from .runtime_sensor_contracts import RuntimeMultimodalSensorSnapshot

_FLIGHT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,119}$")


class RuntimeMultimodalDatasetRecord(StrictModel):
    sample_id: str = Field(pattern=r"^sensor-sample-[0-9a-f]{24}$")
    flight_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=120)
    map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recorded_at_unix_ms: int = Field(ge=0, le=2**63 - 1, strict=True)
    recorded_at_monotonic_seconds: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    rgb_sample_monotonic_seconds: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    rgb_relative_path: str = Field(pattern=r"^rgb/[0-9a-f]{64}\.png$")
    rgb_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_mask_relative_path: str | None = Field(
        default=None,
        pattern=r"^semantic/[0-9a-f]{64}\.png$",
    )
    semantic_mask_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    semantic_label_map_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    semantic_sample_monotonic_seconds: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False, strict=True
    )
    rgb_semantic_time_offset_seconds: float | None = Field(
        default=None, ge=0.0, le=0.1, allow_inf_nan=False, strict=True
    )
    depth_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sensor_snapshot: RuntimeMultimodalSensorSnapshot
    state: dict[str, Any]
    previous_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    # 功能：
    #   验证采样先于记录、图像路径绑定摘要及成套语义监督的时间差，不授予训练资格。
    # 输入：
    #   self：字段类型与范围已经通过校验的记录。
    # 输出：
    #   self：通过跨字段一致性检查的记录。
    @model_validator(mode="after")
    def validate_semantic_binding(self) -> RuntimeMultimodalDatasetRecord:
        values = (
            self.semantic_mask_relative_path,
            self.semantic_mask_sha256,
            self.semantic_label_map_sha256,
            self.semantic_sample_monotonic_seconds,
            self.rgb_semantic_time_offset_seconds,
        )
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError("semantic supervision fields must be supplied together")
        if self.rgb_relative_path != f"rgb/{self.rgb_sha256}.png":
            raise ValueError("multimodal RGB path does not bind its digest")
        if (self.rgb_sample_monotonic_seconds > self.recorded_at_monotonic_seconds
                or self.sensor_snapshot.captured_at_monotonic_seconds
                > self.recorded_at_monotonic_seconds):
            raise ValueError("multimodal sample clock is newer than its recording")
        if self.semantic_sample_monotonic_seconds is not None:
            if self.semantic_mask_relative_path != f"semantic/{self.semantic_mask_sha256}.png":
                raise ValueError("semantic mask path does not bind its digest")
            offset = abs(self.semantic_sample_monotonic_seconds - self.rgb_sample_monotonic_seconds)
            if (self.semantic_sample_monotonic_seconds > self.recorded_at_monotonic_seconds
                    or not math.isclose(offset, self.rgb_semantic_time_offset_seconds,
                                        rel_tol=0.0, abs_tol=1e-12)):
                raise ValueError("semantic supervision clock binding is inconsistent")
        return self


class RuntimeMultimodalDatasetRecorder:
    """Write deduplicated PNGs and an append-only bounded evidence chain."""

    # 功能：
    #   独占建立一次飞行的数据集目录与摘要链；拒绝链接、已有目录和非法容量／频率。
    # 输入：
    #   self：新记录器。
    #   root：必须尚不存在的输出目录。
    #   flight_id、map_sha256：规范飞行标识与实际地图摘要。
    #   maximum_bytes：图像和记录的总写入预算。
    #   minimum_period_seconds：两次接受记录的最短间隔。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, root: Path, *, flight_id: str, map_sha256: str,
                 maximum_bytes: int = 5 * 1024**3, minimum_period_seconds: float = 0.1) -> None:
        if type(maximum_bytes) is not int or not 1024 * 1024 <= maximum_bytes <= 20 * 1024**3:
            raise ValueError("multimodal dataset quota is outside the safe range")
        if (type(minimum_period_seconds) not in (int, float)
                or not 0.05 <= minimum_period_seconds <= 10.0):
            raise ValueError("multimodal recording period is outside the safe range")
        if type(flight_id) is not str or _FLIGHT_ID_PATTERN.fullmatch(flight_id) is None:
            raise ValueError("multimodal flight identity must be normalized lowercase")
        if type(map_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", map_sha256) is None:
            raise ValueError("multimodal map identity must be a SHA-256 digest")
        # 不先 resolve 消除链接痕迹；独占 mkdir 负责已有目标和创建竞争的拒绝。
        self.root = Path(root).absolute()
        check_plain_plugin_path(self.root)
        self.rgb_root = self.root / "rgb"
        self.semantic_root = self.root / "semantic"
        self.records_path = self.root / "records.jsonl"
        self.root.mkdir(parents=True)
        self.rgb_root.mkdir()
        self.semantic_root.mkdir()
        self._directories = {path: path.stat() for path in (
            self.root, self.rgb_root, self.semantic_root
        )}
        self.flight_id = flight_id
        self.map_sha256 = map_sha256
        self.maximum_bytes = maximum_bytes
        self.minimum_period_seconds = minimum_period_seconds
        self._record_count = 0
        self._bytes_written = 0
        self._last_recorded_monotonic_seconds: float | None = None
        self._last_rgb_sample_monotonic_seconds: float | None = None
        self._previous_record_sha256 = "0" * 64
        self._records_digest = hashlib.sha256()
        self._records_metadata: os.stat_result | None = None
        self._records_bytes = 0
        self._asset_sizes: dict[Path, int] = {}
        self._issue: str | None = None
        self._lock = Lock()

    # 功能：
    #   发布完整图像而不覆盖目标；仅删除仍属于本次的暂存文件，异常保留未归属内容。
    # 输入：
    #   self：持有数据集的记录器。
    #   path：必须尚不存在的图像路径。
    #   payload：已经有界校验的不可变编码字节。
    # 输出：
    #   None：不返回业务数据。
    def _atomic_bytes(self, path: Path, payload: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        owned = None
        try:
            check_plain_plugin_path(temporary)
            with temporary.open("xb") as stream:
                owned = os.fstat(stream.fileno())
                if stream.write(payload) != len(payload):
                    raise OSError("MULTIMODAL_DATASET_SHORT_WRITE")
                stream.flush()
                os.fsync(stream.fileno())
            check_plain_plugin_path(temporary)
            check_plain_plugin_path(path)
            if not os.path.samestat(owned, temporary.stat()):
                raise ValueError("MULTIMODAL_DATASET_TEMPORARY_CHANGED")
            os.link(temporary, path)
        finally:
            if owned is not None:
                with suppress(OSError, ValueError):
                    check_plain_plugin_path(temporary)
                    if os.path.samestat(owned, temporary.stat()):
                        temporary.unlink()

    # 功能：
    #   复核自有目录及上次完整记录的身份、长度与修改时间，避免续写被替换的摘要链。
    # 输入：
    #   self：当前记录器。
    # 输出：
    #   None：不返回业务数据。
    def _check_storage(self) -> None:
        for path, owned in self._directories.items():
            check_plain_plugin_path(path)
            if not os.path.samestat(owned, path.stat()):
                raise ValueError("MULTIMODAL_DATASET_DIRECTORY_CHANGED")
        check_plain_plugin_path(self.records_path)
        if self._records_metadata is None:
            if self.records_path.exists():
                raise ValueError("MULTIMODAL_DATASET_RECORDS_CHANGED")
            return
        current = self.records_path.stat()
        if (not os.path.samestat(self._records_metadata, current)
                or current.st_size != self._records_bytes
                or current.st_mtime_ns != self._records_metadata.st_mtime_ns):
            raise ValueError("MULTIMODAL_DATASET_RECORDS_CHANGED")

    # 功能：
    #   1. 冻结有限传感器与状态输入，验证来源时钟和监督配对，再按最小周期采样。
    #   2. 去重图像并追加哈希链；磁盘异常锁定失败，保留现场且不继续扩展不完整记录。
    #   3. 编码字节由上游提供，本层不以文件后缀代替图像解码、训练准入或飞行验收。
    # 输入：
    #   self：仅供本数据集写入的记录器。
    #   rgb_png、frame、sensor_snapshot：RGB 编码、深度帧与传感器状态。
    #   recorded_at_unix_ms、recorded_at_monotonic_seconds：记录开始时刻。
    #   state：需要绑定的有限运行状态字典。
    #   rgb_sample_monotonic_seconds：RGB 原始采样时刻；兼容省略时使用记录时刻。
    #   semantic_mask_png、semantic_label_map_sha256：可选语义掩码及标签定义摘要。
    #   semantic_sample_monotonic_seconds：必须与掩码一并提供的监督采样时刻。
    # 输出：
    #   record：已同步写入的记录；同周期或重复源帧返回 None，不将轮询次数计作新图像。
    def record(self, *, rgb_png: bytes, frame: OnboardPerceptionFrame,
               sensor_snapshot: RuntimeMultimodalSensorSnapshot, recorded_at_unix_ms: int,
               recorded_at_monotonic_seconds: float, state: dict[str, Any],
               rgb_sample_monotonic_seconds: float | None = None,
               semantic_mask_png: bytes | None = None, semantic_label_map_sha256: str | None = None,
               semantic_sample_monotonic_seconds: float | None = None,
               ) -> RuntimeMultimodalDatasetRecord | None:
        if type(rgb_png) is not bytes or not 0 < len(rgb_png) <= 16 * 1024 * 1024:
            raise ValueError("multimodal RGB payload is outside the bounded range")
        semantic_values = (
            semantic_mask_png,
            semantic_label_map_sha256,
            semantic_sample_monotonic_seconds,
        )
        if any(value is not None for value in semantic_values) and not all(
            value is not None for value in semantic_values
        ):
            raise ValueError("semantic supervision payload is incomplete")
        if semantic_mask_png is not None and (
            type(semantic_mask_png) is not bytes or not 0 < len(semantic_mask_png) <= 4 * 1024**2
        ):
            raise ValueError("semantic mask payload is outside the bounded range")
        if semantic_label_map_sha256 is not None and (
            type(semantic_label_map_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", semantic_label_map_sha256) is None
        ):
            raise ValueError("semantic label map identity is invalid")
        rgb_sample_time = (
            recorded_at_monotonic_seconds
            if rgb_sample_monotonic_seconds is None
            else rgb_sample_monotonic_seconds
        )
        clock_values = (recorded_at_monotonic_seconds, rgb_sample_time)
        if semantic_sample_monotonic_seconds is not None:
            clock_values += (semantic_sample_monotonic_seconds,)
        for value in clock_values:
            if (
                type(value) not in (int, float) or not 0 <= value <= 1.7976931348623157e308
            ):
                raise ValueError("MULTIMODAL_DATASET_CLOCK_INVALID")
        if (type(recorded_at_unix_ms) is not int or not 0 <= recorded_at_unix_ms <= 2**63 - 1
                or type(state) is not dict):
            raise ValueError("MULTIMODAL_DATASET_METADATA_INVALID")
        # 在模型 JSON 序列化将 NaN 转为 null 之前，先验证原始数值并冻结可变输入。
        frame = OnboardPerceptionFrame.model_validate(plugin_json_value(frame, limit=16 * 1024**2))
        sensor_snapshot = RuntimeMultimodalSensorSnapshot.model_validate(
            plugin_json_value(sensor_snapshot)
        )
        state = plugin_json_value(state)
        semantic_time_offset = (
            abs(semantic_sample_monotonic_seconds - rgb_sample_time)
            if semantic_sample_monotonic_seconds is not None
            else None
        )
        if semantic_time_offset is not None and semantic_time_offset > 0.1:
            raise ValueError("RGB and semantic samples exceed the synchronization contract")
        with self._lock:
            if self._issue is not None:
                raise RuntimeError("MULTIMODAL_DATASET_FAILED")
            previous_time = self._last_recorded_monotonic_seconds
            if previous_time is not None and recorded_at_monotonic_seconds < previous_time:
                raise ValueError("MULTIMODAL_DATASET_CLOCK_REVERSED")
            if (rgb_sample_time > recorded_at_monotonic_seconds
                    or sensor_snapshot.captured_at_monotonic_seconds > recorded_at_monotonic_seconds
                    or (semantic_sample_monotonic_seconds is not None
                        and semantic_sample_monotonic_seconds > recorded_at_monotonic_seconds)):
                raise ValueError("MULTIMODAL_DATASET_SAMPLE_CLOCK_INVALID")
            previous_rgb = self._last_rgb_sample_monotonic_seconds
            if previous_rgb is not None:
                if rgb_sample_time < previous_rgb:
                    raise ValueError("MULTIMODAL_DATASET_RGB_CLOCK_REVERSED")
                if rgb_sample_time == previous_rgb:
                    return None
            if (
                previous_time is not None
                and recorded_at_monotonic_seconds - previous_time
                < self.minimum_period_seconds
            ):
                return None
            rgb_sha256 = hashlib.sha256(rgb_png).hexdigest()
            rgb_relative_path = f"rgb/{rgb_sha256}.png"
            rgb_path = self.root / rgb_relative_path
            image_growth = 0 if rgb_path in self._asset_sizes else len(rgb_png)
            semantic_sha256 = (
                hashlib.sha256(semantic_mask_png).hexdigest()
                if semantic_mask_png is not None
                else None
            )
            semantic_relative_path = (
                f"semantic/{semantic_sha256}.png"
                if semantic_sha256 is not None
                else None
            )
            semantic_path = (
                self.root / semantic_relative_path
                if semantic_relative_path is not None
                else None
            )
            semantic_growth = (
                0
                if semantic_path is None or semantic_path in self._asset_sizes
                else len(semantic_mask_png or b"")
            )
            base_record = {
                "sample_id": f"sensor-sample-{uuid4().hex[:24]}",
                "flight_id": self.flight_id,
                "map_sha256": self.map_sha256,
                "recorded_at_unix_ms": recorded_at_unix_ms,
                "recorded_at_monotonic_seconds": recorded_at_monotonic_seconds,
                "rgb_sample_monotonic_seconds": rgb_sample_time,
                "rgb_relative_path": rgb_relative_path,
                "rgb_sha256": rgb_sha256,
                "semantic_mask_relative_path": semantic_relative_path,
                "semantic_mask_sha256": semantic_sha256,
                "semantic_label_map_sha256": semantic_label_map_sha256,
                "semantic_sample_monotonic_seconds": semantic_sample_monotonic_seconds,
                "rgb_semantic_time_offset_seconds": semantic_time_offset,
                "depth_frame_sha256": sha256_json(frame),
                "sensor_snapshot": sensor_snapshot.model_dump(mode="json"),
                "state": state,
                "previous_record_sha256": self._previous_record_sha256,
            }
            record = RuntimeMultimodalDatasetRecord.model_validate(
                {**base_record, "record_sha256": "0" * 64}
            )
            # 先规范整数／浮点等类型及默认字段，再散列下游实际回读的表示。
            record_payload = record.model_dump(mode="json")
            record_payload.pop("record_sha256")
            record_sha256 = sha256_json(record_payload)
            record.record_sha256 = record_sha256
            serialized = (record.model_dump_json() + "\n").encode("utf-8")
            # 最新记录消费者只读取有界尾行；不能产生下游无法完整接收的单条记录。
            encode_json(record.model_dump(mode="json"), limit=MAX_MESSAGE_BYTES)
            if len(serialized) > MAX_MESSAGE_BYTES:
                raise ValueError("MULTIMODAL_DATASET_RECORD_TOO_LARGE")
            growth = image_growth + semantic_growth + len(serialized)
            if self._bytes_written + growth > self.maximum_bytes:
                raise RuntimeError("MULTIMODAL_DATASET_QUOTA_EXCEEDED")
            try:
                self._check_storage()
                for path, payload, digest in ((rgb_path, rgb_png, rgb_sha256),
                                             (semantic_path, semantic_mask_png, semantic_sha256)):
                    if path is None:
                        continue
                    if path in self._asset_sizes:
                        observed = read_plugin_file(path, limit=self._asset_sizes[path])
                        if hashlib.sha256(observed).hexdigest() != digest:
                            raise RuntimeError("MULTIMODAL_DATASET_IMAGE_CONTENT_DRIFT")
                    else:
                        self._atomic_bytes(path, payload)
                        # 图像已实际占用磁盘，即使稍后的记录追加失败也不回退此项计数。
                        self._asset_sizes[path] = len(payload)
                        self._bytes_written += len(payload)
                self._check_storage()
                mode = "xb" if self._records_metadata is None else "r+b"
                with self.records_path.open(mode) as handle:
                    opened = os.fstat(handle.fileno())
                    if (not stat.S_ISREG(opened.st_mode)
                            or opened.st_size != self._records_bytes
                            or (self._records_metadata is not None
                                and (not os.path.samestat(self._records_metadata, opened)
                                     or opened.st_mtime_ns != self._records_metadata.st_mtime_ns))):
                        raise ValueError("MULTIMODAL_DATASET_RECORDS_CHANGED")
                    handle.seek(0, os.SEEK_END)
                    if handle.write(serialized) != len(serialized):
                        raise OSError("MULTIMODAL_DATASET_SHORT_WRITE")
                    handle.flush()
                    os.fsync(handle.fileno())
                    after = os.fstat(handle.fileno())
                check_plain_plugin_path(self.records_path)
                current = self.records_path.stat()
                if (not os.path.samestat(after, current)
                        or current.st_mtime_ns != after.st_mtime_ns
                        or current.st_size != after.st_size
                        or after.st_size != self._records_bytes + len(serialized)):
                    raise ValueError("MULTIMODAL_DATASET_RECORDS_CHANGED")
                self._records_metadata = after
            except BaseException as error:
                # 不截断、删除或重试不确定的尾部；由失败回执阻止它被当成完整数据集。
                self._issue = f"MULTIMODAL_DATASET_WRITE_FAILED:{type(error).__name__}"
                raise
            self._records_digest.update(serialized)
            self._records_bytes += len(serialized)
            self._bytes_written += len(serialized)
            self._record_count += 1
            self._last_recorded_monotonic_seconds = recorded_at_monotonic_seconds
            self._last_rgb_sample_monotonic_seconds = rgb_sample_time
            self._previous_record_sha256 = record_sha256
            return record

    # 功能：
    #   用内存计数和增量摘要返回记录状态，不重读增长的文件，不授予训练或飞行资格。
    # 输入：
    #   self：待查询的记录器。
    # 输出：
    #   summary：成功记录计数、已计量字节、摘要及持久化失败标识。
    def summary(self) -> dict[str, object]:
        with self._lock:
            summary = {
                "flight_id": self.flight_id,
                "map_sha256": self.map_sha256,
                "record_count": self._record_count,
                "bytes_written": self._bytes_written,
                "maximum_bytes": self.maximum_bytes,
                "minimum_period_seconds": self.minimum_period_seconds,
                "latest_record_sha256": self._previous_record_sha256,
                "records_sha256": self._records_digest.copy().hexdigest(),
                "qualification_granted": False,
                "issue_code": self._issue,
                "bytes_accounting_complete": self._issue is None,
            }
            return summary
