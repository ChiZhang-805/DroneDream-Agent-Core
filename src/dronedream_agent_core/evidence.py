"""Append-only hash-linked evidence records for a single mission run."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from dronedream_plugin_sdk.protocol import copy_json, decode_json, encode_json

from .contracts import EvidenceRecord
from .hashing import sha256_json
from .plugin_files import check_plain_plugin_path, read_plugin_file

ZERO_HASH = "0" * 64
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_CHAIN_BYTES = 64 * 1024 * 1024


class EvidenceChain:
    """Hash-link a run's records; the caller must serialize writes to this path.

    Hashes reveal inconsistent edits, not authenticity: an external trusted
    anchor is needed to detect wholesale replacement or removal of the tail.
    """

    # 功能：
    #   建立单任务证据链入口，不截断旧记录；同实例线程串行，跨进程仍须单一写入者。
    # 输入：
    #   path：任务独立证据文件路径，不能经过符号链接或重解析点。
    # 输出：
    #   self：持有路径和线程锁的证据链对象。
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        check_plain_plugin_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    # 功能：
    #   有界读取确切文件快照，拒绝重复 JSON 键、非法数值、半行和损坏的散列链接。
    # 输入：
    #   self：调用方已持有当前实例锁。
    # 输出：
    #   snapshot：已核验记录列表与其原始字节二元组。
    def _read_snapshot(self) -> tuple[list[EvidenceRecord], bytes]:
        check_plain_plugin_path(self.path)
        try:
            content = read_plugin_file(self.path, limit=MAX_CHAIN_BYTES)
        except FileNotFoundError:
            content = b""
        if content and not content.endswith(b"\n"):
            raise ValueError("evidence final record is incomplete")
        records = []
        for line in content.splitlines():
            if line.strip():
                parsed = decode_json(line, limit=MAX_RECORD_BYTES)
                record = EvidenceRecord.model_validate_json(
                    encode_json(parsed, limit=MAX_RECORD_BYTES), strict=True
                )
                records.append(record)
        self.verify(records)
        snapshot = records, content
        return snapshot

    # 功能：
    #   仅暴露整条链验证通过的记录，不把散列一致当作作者真实性或完整飞行证明。
    # 输入：
    #   self：当前证据链入口。
    # 输出：
    #   records：独立解析并核验后的记录列表。
    def read(self) -> list[EvidenceRecord]:
        with self._lock:
            records, _ = self._read_snapshot()
        return records

    # 功能：
    #   冻结事件载荷、核验既有链并复核写入前缀，再追加和 fsync；错误不静默重建旧证据。
    #   此方法需读取既有记录，不属于高频飞控路径；跨进程调用须由任务所有者串行。
    # 输入：
    #   event_type：事件类型。
    #   payload：标准有限 JSON 数据，不能含会被字符串化而冲突的字典键。
    # 输出：
    #   record：已写入、与原输入容器分离的新证据记录。
    def append(self, event_type: str, payload: dict[str, Any]) -> EvidenceRecord:
        payload = copy_json(payload, limit=MAX_RECORD_BYTES)
        with self._lock:
            records, content = self._read_snapshot()
            previous = records[-1].record_sha256 if records else ZERO_HASH
            record = EvidenceRecord(
                sequence=len(records) + 1,
                created_at=datetime.now(UTC),
                event_type=event_type,
                artifact_sha256=sha256_json(payload),
                previous_record_sha256=previous,
                record_sha256=ZERO_HASH,
                payload=payload,
            )
            body = record.model_dump(mode="json", exclude={"record_sha256"})
            record.record_sha256 = sha256_json(body)
            encoded = encode_json(record.model_dump(mode="json"), limit=MAX_RECORD_BYTES)
            row = encoded.encode("utf-8") + b"\n"
            if len(content) + len(row) > MAX_CHAIN_BYTES:
                raise ValueError("evidence chain capacity exceeded")
            check_plain_plugin_path(self.path)
            with self.path.open("a+b") as handle:
                handle.seek(0)
                # 以实际打开文件的原文字节复核，不能只核对大小或重新读取后的最后一个摘要。
                if handle.read(len(content) + 1) != content:
                    raise ValueError("evidence prefix changed before append")
                check_plain_plugin_path(self.path)
                if not os.path.samestat(os.fstat(handle.fileno()), self.path.stat()):
                    raise ValueError("evidence file replaced before append")
                handle.seek(0, os.SEEK_END)
                if handle.write(row) != len(row):
                    raise OSError("evidence append incomplete")
                handle.flush()
                os.fsync(handle.fileno())
        return record

    # 功能：
    #   重新核验每条记录的类型、序列、前驱链接、记录摘要与独立载荷摘要。
    #   全链替换或尾部删除仍需外部可信锚点发现，单靠本地散列不能证明真实性。
    # 输入：
    #   records：待核验的证据记录序列。
    # 输出：
    #   None：全部成立时正常返回，不一致立即抛出异常。
    @staticmethod
    def verify(records: list[EvidenceRecord]) -> None:
        previous = ZERO_HASH
        for expected_sequence, record in enumerate(records, start=1):
            record = EvidenceRecord.model_validate_json(
                encode_json(record.model_dump(mode="json"), limit=MAX_RECORD_BYTES), strict=True
            )
            if record.sequence != expected_sequence:
                raise ValueError("evidence sequence is discontinuous")
            if record.previous_record_sha256 != previous:
                raise ValueError("evidence previous hash does not match")
            body = record.model_dump(mode="json", exclude={"record_sha256"})
            if sha256_json(body) != record.record_sha256:
                raise ValueError("evidence record hash does not match content")
            if record.artifact_sha256 != sha256_json(record.payload):
                raise ValueError("evidence artifact hash does not match payload")
            previous = record.record_sha256

    # 功能：
    #   仅导出验证通过的记录，不修改追加式源文件。
    # 输入：
    #   self：当前证据链入口。
    # 输出：
    #   exported：便于查看的完整 JSON 文本。
    def export_json(self) -> str:
        exported = json.dumps(
            [record.model_dump(mode="json") for record in self.read()],
            ensure_ascii=False,
            indent=2,
        )
        return exported
