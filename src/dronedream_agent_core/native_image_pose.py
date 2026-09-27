"""Source-clock image poses for an explicitly bound simulation/vehicle clock.

The caller establishes the shared clock; this module never infers it from
timestamp magnitude. No map fitting, truth, covariance reduction or commands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dronedream_plugin_sdk.protocol import copy_json

from .hashing import sha256_json
from .localization_evidence import position_variance_bound_m2
from .native_odometry_source import (
    SourceOdometryHistory,
    _integer,
    _real,
    canonical_odometry_fields,
    decode_source_odometry,
    source_pose_in_map,
)
from .native_pose import NativeMapPose


@dataclass(frozen=True)
class SourceImagePose:
    pose: NativeMapPose
    source_skew_us: int
    conservative_variance_m2: float
    source_evidence: dict


class NativeImagePoseBuffer:
    """Single-owner buffer; caller holds its existing native-state lock."""

    # 功能：创建明确时钟域内的有界图像位姿缓存，不推断物理设备和仿真时钟一致。
    # 输入：clock_domain：启动器核验的本次运行时钟身份。
    # 输出：空缓存；尚无有效姿态。
    def __init__(self, *, clock_domain):
        if type(clock_domain) is not str or not 1 <= len(clock_domain) <= 160:
            raise ValueError("IMAGE_SOURCE_CLOCK_DOMAIN_INVALID")
        self.clock_domain = clock_domain
        self._history = None
        self._records = {}
        self._binding = None
        self._binding_hash = None
        self._invalidated = False

    # 功能：保留完整单包姿态及其首次接收/消费时刻，校验重复字段与固定地图绑定。
    # 输入：payload：原生遥测；available_at_monotonic：本消费者首次见到消息的时刻。
    # 输出：无；消息损坏后该缓存失效，不能借用旧数据回退到接收时间配对。
    def ingest(self, payload, *, available_at_monotonic):
        if self._invalidated:
            raise ValueError("IMAGE_SOURCE_HISTORY_INVALIDATED")
        try:
            self._ingest(payload, available_at_monotonic=available_at_monotonic)
        except (ValueError, TypeError, KeyError):
            self._records.clear()
            self._invalidated = True
            raise

    # 功能：在唯一写入线程中校验来源，按原始时间存储最多 128 条不可变协议消息。
    # 输入：payload、available_at_monotonic：与公开接入口相同。
    # 输出：无；不更新主控制状态，也不修改原生协方差。
    def _ingest(self, payload, *, available_at_monotonic):
        odometry = payload["dynamics"]["sources"]["odometry"]
        source = odometry["mavlink_source"]
        packet = decode_source_odometry(
            source["fields_json"],
            system_id=source["system_id"],
            component_id=source["component_id"],
            received_monotonic=source["received_monotonic_seconds"],
        )
        received_ms = _integer(odometry["received_at_unix_ms"], 0, 2**63 - 1)
        if (
            type(odometry["timestamp_us"]) is not int
            or odometry["timestamp_us"] != packet.timestamp_us
            or type(odometry["reset_counter"]) is not int
            or odometry["reset_counter"] != packet.reset_counter
        ):
            raise ValueError("IMAGE_SOURCE_PACKET_IDENTITY_MISMATCH")
        expected = canonical_odometry_fields(packet)
        # Hash comparison is type-sensitive: True must not equal 1.0 inside a
        # covariance array. Both consumers must see precisely the same packet.
        if sha256_json({key: odometry.get(key) for key in expected}) != sha256_json(expected):
            raise ValueError("IMAGE_SOURCE_CANONICAL_FIELDS_MISMATCH")
        binding = copy_json(payload["map_frame_binding"], limit=16384)
        binding_hash = payload["map_frame_binding_sha256"]
        # Also verifies the raw coordinate IDs, quaternion and fixed origin.
        source_pose_in_map(packet, binding=binding, binding_sha256=binding_hash)
        if self._binding_hash is not None and binding_hash != self._binding_hash:
            raise ValueError("IMAGE_SOURCE_MAP_BINDING_CHANGED")
        if self._history is None:
            self._history = SourceOdometryHistory(
                clock_domain=self.clock_domain,
                system_id=packet.system_id,
                component_id=packet.component_id,
                capacity=128,
            )
            self._binding, self._binding_hash = binding, binding_hash
        new = self._history.ingest(packet, available_at_monotonic=available_at_monotonic)
        if new:
            active = {item.timestamp_us for item in self._history._history}
            self._records = {
                stamp: value for stamp, value in self._records.items() if stamp in active
            }
            self._records[packet.timestamp_us] = (received_ms, source["fields_json"])

    # 功能：用实际图像源时刻选择同段完整位姿，分别检查主机新鲜度和源时差预算。
    # 输入：图像源纳秒、时钟域、原接收时刻、消费时刻，以及量程和加速度上限。
    # 输出：源时钟对齐位姿与增加而非缩小的方差；没有位姿外推或定位资格声明。
    def align(
        self,
        *,
        image_timestamp_ns,
        clock_domain,
        image_received_unix_ms,
        now_unix_ms,
        now_monotonic,
        maximum_range_m,
        maximum_acceleration_mps2,
    ):
        _integer(image_timestamp_ns, 0, 2**63 - 1)
        _integer(image_received_unix_ms, 0, 2**63 - 1)
        _integer(now_unix_ms, 0, 2**63 - 1)
        if image_timestamp_ns % 1000:
            raise ValueError("IMAGE_SOURCE_MICROSECOND_STAMP_REQUIRED")
        if self._invalidated or self._history is None:
            raise ValueError("IMAGE_SOURCE_HISTORY_UNAVAILABLE")
        if not 0 <= now_unix_ms - image_received_unix_ms <= 250:
            raise ValueError("IMAGE_SOURCE_RECEIPT_EXPIRED")
        maximum_range_m = _real(maximum_range_m)
        maximum_acceleration_mps2 = _real(maximum_acceleration_mps2)
        if not 0 < maximum_range_m <= 1000 or not 0 < maximum_acceleration_mps2 <= 100:
            raise ValueError("IMAGE_SOURCE_ALIGNMENT_LIMIT_INVALID")
        packet = self._history.align(
            timestamp_us=image_timestamp_ns // 1000,
            clock_domain=clock_domain,
            available_at_monotonic=now_monotonic,
        )
        received_ms, fields_json = self._records[packet.timestamp_us]
        if not 0 <= now_unix_ms - received_ms <= 250:
            raise ValueError("IMAGE_SOURCE_ODOMETRY_EXPIRED")
        position, orientation, velocity = source_pose_in_map(
            packet, binding=self._binding, binding_sha256=self._binding_hash
        )
        source_skew_us = packet.timestamp_us - image_timestamp_ns // 1000
        dt = abs(source_skew_us) / 1_000_000
        # Norms are invariant under the supported orthonormal child frames.
        displacement = math.hypot(*packet.velocity_child_frame_m_s) * dt
        displacement += 0.5 * maximum_acceleration_mps2 * dt * dt
        omega = math.hypot(*packet.angular_velocity_child_frame_rad_s)
        displacement += maximum_range_m * (2 * math.sin(min(math.pi, omega * dt) / 2))
        variance = (
            math.sqrt(position_variance_bound_m2(packet.pose_covariance_upper)) + displacement
        ) ** 2
        if not math.isfinite(variance):
            raise ValueError("IMAGE_SOURCE_VARIANCE_INVALID")
        pose = NativeMapPose(
            position,
            orientation,
            velocity,
            received_ms,
            self._binding_hash,
            received_ms,
            received_ms,
        )
        evidence = {
            "clock_domain": self.clock_domain,
            "image_timestamp_ns": image_timestamp_ns,
            "source_skew_us": source_skew_us,
            "reset_counter": packet.reset_counter,
            "odometry_timestamp_us": packet.timestamp_us,
            "received_at_unix_ms": received_ms,
            "received_monotonic_seconds": packet.received_monotonic_seconds,
            "system_id": packet.system_id,
            "component_id": packet.component_id,
            "fields_json": fields_json,
            "raw_fields_sha256": sha256_json(fields_json),
            "qualification_granted": False,
            "pose_extrapolated": False,
        }
        return SourceImagePose(pose, source_skew_us, variance, evidence)
