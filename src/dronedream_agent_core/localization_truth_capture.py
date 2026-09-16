"""Bounded independent pose witness; never consumed by the flight controller."""

from __future__ import annotations

import json
import math
from pathlib import Path

from dronedream_plugin_sdk.protocol import copy_json, encode_json

from .hashing import sha256_json
from .runtime_evidence import BoundedRuntimeEvidenceWriter, reserve_new_evidence_files
from .simulation_sensor_frames import (
    collision_center_from_canonical,
    select_canonical_poses,
    simulation_pose_time_ns,
)

MAXIMUM_TRUTH_RECORDS = 4096
MAXIMUM_TRUTH_RECORD_BYTES = 4096


# 功能：
#   在后台验证见证记录的有限 JSON 类型及预算，再对全部原始字段计算摘要。
# 输入：
#   record：未带自身摘要的独立仿真见证记录。
# 输出：
#   text：带摘要的 JSON 文本，连同换行不超过四 KiB。
def _serialize(record):
    if type(record) is not dict or "record_sha256" in record:
        raise ValueError("LOCALIZATION_TRUTH_RECORD_INVALID")
    encode_json(record, limit=MAXIMUM_TRUTH_RECORD_BYTES - 1)
    text = json.dumps({**record, "record_sha256": sha256_json(record)},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text.encode("utf-8")) + 1 > MAXIMUM_TRUTH_RECORD_BYTES:
        raise ValueError("LOCALIZATION_TRUTH_RECORD_TOO_LARGE")
    return text


class LocalizationTruthCapture:
    """Finite 50 Hz maximum witness clip, separate from native sensor capture."""

    # 功能：
    #   1. 验证并冻结安装契约和几何，独占领取本次离线见证文件。
    #   2. 契约摘要只证明内容一致；真值流始终不作为控制输入或飞行授权。
    # 输入：
    #   self：待初始化的独立见证采集器。
    #   run_dir：本次运行的私有目录。
    #   frames：已经从安装模型解析的坐标契约。
    #   summary_publisher：发布完整回执的回调。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, run_dir: Path, *, frames: dict, summary_publisher):
        if not callable(summary_publisher):
            raise ValueError("LOCALIZATION_TRUTH_PUBLISHER_INVALID")
        if type(frames) is not dict:
            raise ValueError("LOCALIZATION_TRUTH_FRAME_CONTRACT_INVALID")
        frames = copy_json(frames, limit=65536)
        content = {k: v for k, v in frames.items() if k != "record_sha256"}
        if (frames.get("record_sha256") != sha256_json(content)
                or frames.get("depth_mount_verified") is not True
                or frames.get("truth_used_as_control_input") is not False):
            raise ValueError("LOCALIZATION_TRUTH_FRAME_CONTRACT_INVALID")
        try:
            names = frames["vehicle_model_name"], frames["canonical_link_name"]
            if any(type(name) is not str or not name.strip() or len(name) > 160 for name in names):
                raise ValueError("frame name invalid")
            # 空实体集合仅验证名称契约；独立几何计算会检查向量和单位四元数。
            select_canonical_poses([], vehicle_name=names[0], canonical_link_name=names[1])
            identity = {"position_m": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0]}
            collision_center_from_canonical(model_world=identity,
                canonical_in_model=frames["canonical_at_rest"],
                canonical_at_rest=frames["canonical_at_rest"],
                collision_center_model_m=frames["collision_center_model_m"])
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise ValueError("LOCALIZATION_TRUTH_FRAME_CONTRACT_INVALID") from error
        self.path = run_dir / "localization-truth-observations.jsonl"
        self.summary_path = run_dir / "localization-truth-summary.json"
        if self.path.exists() or self.summary_path.exists():
            raise FileExistsError("LOCALIZATION_TRUTH_REQUIRES_NEW_RUN")
        self.frames = frames
        reserve_new_evidence_files(self.path, self.summary_path)
        self._publish = summary_publisher
        self._last_time = self._last_scene = self._last_unix = None
        self._count = self._skipped = 0
        self._issue = None
        self._closed = False
        self._writer = BoundedRuntimeEvidenceWriter(self.summary_path,
            summary_publisher=self._publish_summary, serializer=_serialize,
            maximum_pending_records=16)

    # 功能：
    #   采样精确绑定实体的独立仿真位姿，拒绝时钟回退并保持与控制估计输入分离。
    # 输入：
    #   self：单生产者使用的见证采集器。
    #   message：包含模型、机体位姿及仿真时刻的消息。
    #   received_monotonic：消息接收的本地单调钟秒数。
    #   received_unix_ms：消息接收的非负整数 UNIX 毫秒数。
    # 输出：
    #   accepted：成功排入记录队列时为 True，尚不代表写盘完成。
    def record(self, message, *, received_monotonic: float, received_unix_ms: int):
        if self._closed or self._issue or self._writer.issue:
            return False
        if self._count >= MAXIMUM_TRUTH_RECORDS:
            self._skipped += 1
            return False
        try:
            if (type(received_monotonic) not in (int, float)
                    or not 0 <= received_monotonic <= 2**53
                    or not math.isfinite(received_monotonic)
                    or type(received_unix_ms) is not int or not 0 <= received_unix_ms < 2**63):
                raise ValueError("LOCALIZATION_TRUTH_CLOCK_INVALID")
            scene = simulation_pose_time_ns(message)
            if self._last_time is not None:
                if (received_monotonic < self._last_time or scene < self._last_scene
                        or received_unix_ms < self._last_unix):
                    raise ValueError("LOCALIZATION_TRUTH_CLOCK_REGRESSED")
                if scene == self._last_scene or received_monotonic - self._last_time < .02:
                    self._skipped += 1
                    return False
            selected = select_canonical_poses(message.pose,
                vehicle_name=self.frames["vehicle_model_name"],
                canonical_link_name=self.frames["canonical_link_name"])
            if selected is None:
                return False  # 场景启动但尚无目标实体，不虚构零位姿。
            model, link, _, _ = selected
            center = collision_center_from_canonical(model_world=model, canonical_in_model=link,
                canonical_at_rest=self.frames["canonical_at_rest"],
                collision_center_model_m=self.frames["collision_center_model_m"])
            record = {"schema_version": "dronedream.localization-truth-observation.v1",
                "sequence": self._count + 1, "frames_sha256": self.frames["record_sha256"],
                "received_monotonic_seconds": received_monotonic,
                "received_at_unix_ms": received_unix_ms, "publisher_simulation_time_ns": scene,
                "model_world": model, "canonical_in_model": link,
                "collision_center_world_enu_m": center, "qualification_granted": False,
                "truth_used_as_control_input": False}
            if not self._writer.submit(self.path, record):
                raise ValueError("LOCALIZATION_TRUTH_ENQUEUE_FAILED")
            self._last_time, self._last_scene, self._last_unix = (
                received_monotonic, scene, received_unix_ms)
            self._count += 1
            accepted = True
            return accepted
        except (ValueError, TypeError, AttributeError, KeyError, OverflowError) as exc:
            self._issue = "LOCALIZATION_TRUTH_INPUT_INVALID:" + type(exc).__name__
            return False

    # 功能：
    #   合并采集错误与后台写入状态，明确真值仅用于独立离线对照。
    # 输入：
    #   self：当前见证采集器。
    #   writer_summary：队列返回的独立状态快照。
    # 输出：
    #   summary：包含失败、配额、计数和权限限制的完整回执。
    def _summary(self, writer_summary):
        summary = {**writer_summary, "capture_issue": self._issue,
            "complete": writer_summary["complete"] and self._issue is None,
            "accepted_count": self._count, "sampling_or_quota_skipped": self._skipped,
            "maximum_records": MAXIMUM_TRUTH_RECORDS,
            "maximum_jsonl_bytes": MAXIMUM_TRUTH_RECORDS * MAXIMUM_TRUTH_RECORD_BYTES,
            "qualification_granted": False, "truth_used_as_control_input": False}
        return summary

    # 功能：
    #   在首次及关闭发布前统一加入采集状态，避免读者观察到短暂的错误成功声明。
    # 输入：
    #   self：当前见证采集器。
    #   path：本次回执路径。
    #   writer_summary：队列原始状态。
    # 输出：
    #   None：不返回业务数据。
    def _publish_summary(self, path, writer_summary):
        self._publish(path, self._summary(writer_summary))

    # 功能：
    #   停止接收并限时排空，完整保留发布错误和超时，不改变控制器状态。
    # 输入：
    #   self：当前见证采集器。
    # 输出：
    #   summary：含采集状态及后台关闭结果的回执。
    def close(self):
        self._closed = True
        summary = self._summary(self._writer.close(timeout_seconds=4.))
        return summary
