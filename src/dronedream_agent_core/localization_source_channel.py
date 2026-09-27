"""Bounded lossless live geometry transport, independent of sampled evidence logs.

The channel carries measurements, not pose qualification or motion authority.
Original source and host receipt clocks are retained; delivery never renews them.
"""

from __future__ import annotations

import base64
import hashlib
import zlib

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .local_packet_channel import PacketContract

MAX_RAW_BYTES = 128 * 1024
MAX_ENCODED_BYTES = 56_000
SCHEMA = "dronedream.localization-source.v1"


# 功能：统一核对原始测量的权限、结构及来源钟，出站不必压缩后再解压回读一次。
# 输入：record：原始测量对象；stamp：报文声明的原始毫秒时刻。
# 输出：无；不完整或改写时刻的记录拒绝，完整几何校验仍由测量消费者承担。
def _validate_record(record, stamp):
    if (type(stamp) is not int or not 0 <= stamp <= 2**63 - 1
            or type(record) is not dict
            or record.get("schema_version") != "dronedream.live-geometry-observation.v1"
            or record.get("motion_permission_granted") is not False
            or record.get("truth_correction_applied") is not False
            or type(record.get("scan")) is not dict
            or type(record["scan"].get("observed_at_unix_ms")) is not int
            or record["scan"]["observed_at_unix_ms"] != stamp):
        raise ValueError("LOCALIZATION_SOURCE_RECORD_INVALID")


# 功能：核对出站测量信封预算，避免刚压缩后又在感知线程解压全部射线。
# 输入：已编码信封；输出：无。只允许传输，不代表原始观测已被接收端验证。
def validate_localization_source_envelope(payload: dict) -> None:
    if (type(payload) is not dict
            or set(payload) != {"schema_version", "updated_at_unix_ms", "sha256", "data"}
            or payload.get("schema_version") != SCHEMA
            or type(payload.get("updated_at_unix_ms")) is not int
            or not 0 <= payload["updated_at_unix_ms"] <= 2**63 - 1
            or type(payload.get("data")) is not str
            or not 1 <= len(payload["data"]) <= MAX_ENCODED_BYTES
            or type(payload.get("sha256")) is not str
            or len(payload["sha256"]) != 64
            or any(c not in '0123456789abcdef' for c in payload['sha256'])):
        raise ValueError("LOCALIZATION_SOURCE_PACKET_INVALID")


# 功能：接收端仍完整有界解压并核验摘要、来源和权限，拒绝截断及尾随内容。
# 输入：经过本次运行认证通道的信封；输出：已验证原始观测，不续期、不授予运动权限。
def decode_localization_source(payload: dict) -> dict:
    validate_localization_source_envelope(payload)
    try:
        compressed = base64.b64decode(payload["data"], validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, MAX_RAW_BYTES + 1)
        if (len(raw) > MAX_RAW_BYTES or not decoder.eof
                or decoder.unused_data or decoder.unconsumed_tail):
            raise ValueError("LOCALIZATION_SOURCE_COMPRESSION_INVALID")
        if hashlib.sha256(raw).hexdigest() != payload["sha256"]:
            raise ValueError("LOCALIZATION_SOURCE_DIGEST_MISMATCH")
        record = decode_json(raw, limit=MAX_RAW_BYTES)
        _validate_record(record, payload["updated_at_unix_ms"])
        return record
    except (zlib.error, UnicodeError) as error:
        raise ValueError("LOCALIZATION_SOURCE_COMPRESSION_INVALID") from error


# 功能：将单帧真实测量无损压缩成固定预算报文；不排队重传、不改变原始来源时刻。
# 输入：record：含扫描、安装标定和原生姿态来源的独立快照。
# 输出：payload：可发送报文；不可压缩的超预算输入直接拒绝。
def encode_localization_source(record: dict) -> dict:
    if (type(record) is not dict or type(record.get("scan")) is not dict
            or type(record["scan"].get("observed_at_unix_ms")) is not int):
        raise ValueError("LOCALIZATION_SOURCE_RECORD_INVALID")
    _validate_record(record, record["scan"]["observed_at_unix_ms"])
    raw = encode_json(record, limit=MAX_RAW_BYTES).encode("utf-8")
    payload = {
        "schema_version": SCHEMA,
        "updated_at_unix_ms": record["scan"]["observed_at_unix_ms"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(zlib.compress(raw, level=1)).decode("ascii"),
    }
    if not 1 <= len(payload["data"]) <= MAX_ENCODED_BYTES:
        raise ValueError("LOCALIZATION_SOURCE_PACKET_INVALID")
    return payload


LOCALIZATION_SOURCE_CONTRACT = PacketContract(
    protocol=SCHEMA, prefix="dd-localization-", payload_schema=SCHEMA,
    validate=decode_localization_source,
    validate_outbound=validate_localization_source_envelope,
)
