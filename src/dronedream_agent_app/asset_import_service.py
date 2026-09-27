"""Resumable, quarantined admission for normalized simulation asset packages."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import threading
import zipfile
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_hex
from uuid import uuid4

from dronedream_agent_core.asset_package_storage import (
    copy_asset_source,
    export_stored_asset,
    extract_verified_asset,
    publish_asset_directory,
    verify_stored_asset,
)
from dronedream_agent_core.asset_packages import (
    MAX_ASSET_IR_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_PACKAGE_ARCHIVE_BYTES,
    AssetImportJob,
    AssetKind,
    InspectedDDPkg,
    inspect_ddpkg,
    normalized_member_path,
    open_verified_ddpkg,
    qualification_is_current,
    transition_import_job,
)
from dronedream_agent_core.asset_pair_qualification import (
    AssetPairQualificationJob,
    AssetPairQualificationReceipt,
)
from dronedream_agent_core.asset_source_adapters import (
    AssetSourceDetection,
    detect_asset_source,
    normalize_asset_source,
)
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_plugin_sdk.protocol import decode_json

from .storage import AppStore, AssetImportError

_TERMINAL_STATES = {"qualified", "failed", "cancelled"}
_QUALIFICATION_REQUIRED_INPUTS = {
    "qualification_evidence",
    "qualification_environment_versions",
    "local_qualification_run",
}
_PROCESSABLE_STATES = {
    "quarantining",
    "parsing",
    "needs_input",
    "normalizing",
    "building",
    "validating",
}
_IMPORT_LOCKS = tuple(threading.RLock() for _ in range(64))


class AssetImportService:
    """Own the quarantine, inspection and atomic content-version promotion boundary."""

    # 功能：
    #   配置隔离导入依赖，并为同一存储的多个服务实例共享进程内状态锁。
    # 输入：
    #   self：待初始化的导入服务。
    #   store：本地资产存储。
    #   environment_versions_provider：当前运行环境标识读取器。
    #   source_detector：原始格式检测器。
    #   plugin_source_normalizer：可选的隔离伴随工具入口。
    #   local_qualification_verifier：可选的本地验收结果核查器。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        store: AppStore,
        *,
        environment_versions_provider: Callable[[], dict[str, str]] | None = None,
        source_detector: Callable[[Path, str], AssetSourceDetection] | None = None,
        plugin_source_normalizer: Callable[..., Path | None] | None = None,
        local_qualification_verifier: (
            Callable[[InspectedDDPkg, dict[str, str]], bool] | None
        ) = None,
    ) -> None:
        self.store = store
        self.environment_versions_provider = (
            (lambda: {}) if environment_versions_provider is None else environment_versions_provider
        )
        self.source_detector = detect_asset_source if source_detector is None else source_detector
        self.plugin_source_normalizer = plugin_source_normalizer
        self.local_qualification_verifier = local_qualification_verifier
        # 固定锁条带不会随任务数量增长；重复创建服务也不能绕过同一存储的串行保护。
        key = str(store.asset_versions_root.resolve()).casefold().encode()
        self._state_lock = _IMPORT_LOCKS[int.from_bytes(hashlib.sha256(key).digest()[:2]) % 64]

    # 功能：
    #   1. 将有界原始文件复制到本次独占隔离目录，按实际复制字节记录摘要。
    #   2. 创建可恢复任务；复制失败时保留隔离现场，不清理其他流程的文件。
    # 输入：
    #   self：当前导入服务。
    #   source：待隔离的普通文件路径。
    #   source_name：用户展示名称，不用于决定存储目录。
    #   source_format：声明的输入格式。
    #   expected_kind：可选的预期资产类别。
    #   owner_id：任务所属本地用户标识。
    # 输出：
    #   record：进入解析状态后的持久任务记录。
    def create(
        self,
        *,
        source: Path,
        source_name: str,
        source_format: str,
        expected_kind: AssetKind | None = None,
        owner_id: str = "desktop-local-user",
    ) -> dict[str, object]:
        if not isinstance(source_name, str):
            raise AssetImportError("ASSET_IMPORT_SOURCE_NAME_INVALID")
        try:
            check_plain_plugin_path(source)
            source_metadata = source.stat()
        except (OSError, ValueError) as error:
            raise AssetImportError("ASSET_IMPORT_SOURCE_INVALID") from error
        if (
            not stat.S_ISREG(source_metadata.st_mode)
            or not 0 < source_metadata.st_size <= MAX_PACKAGE_ARCHIVE_BYTES
        ):
            raise AssetImportError("ASSET_IMPORT_SOURCE_SIZE_OR_TYPE_INVALID")
        display_name = Path(source_name.replace("\\", "/")).name
        if not display_name or display_name in {".", ".."}:
            raise AssetImportError("ASSET_IMPORT_SOURCE_NAME_INVALID")
        now = datetime.now(UTC)
        job = AssetImportJob(
            job_id=f"asset-job-{token_hex(12)}",
            owner_id=owner_id,
            source_name=display_name,
            source_format=source_format,
            asset_kind=expected_kind,
            progress_percent=0,
            revision=0,
            created_at=now,
            updated_at=now,
        )
        destination_directory = self.store.asset_quarantine_root / job.job_id
        destination_directory.mkdir(parents=False)
        lowered_name = display_name.casefold()
        suffix = (
            ".building.yaml"
            if lowered_name.endswith(".building.yaml")
            else Path(display_name).suffix.casefold()
        )
        if (
            not suffix
            or len(suffix) > 16
            or any(character not in ".abcdefghijklmnopqrstuvwxyz0123456789" for character in suffix)
        ):
            suffix = ".upload"
        destination = destination_directory / f"source{suffix}"
        job = transition_import_job(job, "quarantining", progress_percent=5)
        try:
            digest = hash_plugin_file(
                source, limit=MAX_PACKAGE_ARCHIVE_BYTES, destination=destination
            )
            job = AssetImportJob.model_validate(
                {**job.model_dump(mode="python"), "package_sha256": digest}
            )
            self.store.save_asset_import_job(
                job,
                source_format=source_format,
                source_path=destination,
                package_sha256=job.package_sha256,
            )
            job = transition_import_job(job, "parsing", progress_percent=15)
            record = self.store.save_asset_import_job(job)
            return record
        except Exception as error:
            # 本次目录已独占创建。保留部分复制及错误记录，避免按“路径存在”推断删除所有权。
            failed = transition_import_job(
                job,
                "failed",
                progress_percent=job.progress_percent,
                issue_codes=[self._public_issue_code(error)],
            )
            with suppress(Exception):
                self.store.save_asset_import_job(
                    failed, source_path=destination, source_format=source_format
                )
            raise AssetImportError(self._public_issue_code(error)) from error

    # 功能：
    #   在同一存储的串行状态保护下恢复或推进导入任务。
    # 输入：
    #   self：当前导入服务。
    #   job_id：需要推进的持久任务标识。
    # 输出：
    #   record：推进后重新保存的任务记录。
    def process(self, job_id: str) -> dict[str, object]:
        with self._state_lock:
            record = self._process_locked(job_id)
            return record

    # 功能：
    #   1. 接收与原始摘要和适配器一致的伴随工具结果，仅将其作为数据解析。
    #   2. 拒绝外部自授资格，复核复制内容后无覆盖发布，再恢复原导入任务。
    # 输入：
    #   self：当前导入服务。
    #   job_id：等待伴随转换结果的任务。
    #   result：伴随工具生成的 DDPKG。
    #   source_package_sha256：工具使用的原始输入摘要。
    #   adapter_id：执行转换的适配器标识。
    # 输出：
    #   record：接收结果并推进后的任务记录。
    def submit_companion_result(
        self,
        job_id: str,
        *,
        result: Path,
        source_package_sha256: str,
        adapter_id: str,
    ) -> dict[str, object]:
        with self._state_lock:
            job = self.store.load_asset_import_job(job_id)
            if job.state != "needs_input":
                raise AssetImportError(f"ASSET_COMPANION_RESULT_NOT_EXPECTED:{job.state}")
            if not job.required_inputs or set(job.required_inputs).issubset(
                _QUALIFICATION_REQUIRED_INPUTS
            ):
                raise AssetImportError("ASSET_COMPANION_RESULT_NOT_EXPECTED")
            if job.package_sha256 != source_package_sha256:
                raise AssetImportError("ASSET_COMPANION_SOURCE_HASH_MISMATCH")
            if job.source_adapter_id is None or job.source_adapter_id != adapter_id:
                raise AssetImportError("ASSET_COMPANION_ADAPTER_MISMATCH")
            source = self.store.get_asset_import_source(job_id)
            if self._sha256(source) != source_package_sha256:
                raise AssetImportError("ASSET_IMPORT_SOURCE_HASH_MISMATCH")
            if not result.is_file() or result.stat().st_size <= 0:
                raise AssetImportError("ASSET_COMPANION_RESULT_INVALID")
            if result.stat().st_size > MAX_PACKAGE_ARCHIVE_BYTES:
                raise AssetImportError("ASSET_COMPANION_RESULT_TOO_LARGE")
            inspected = inspect_ddpkg(result)
            if inspected.manifest.qualification is not None:
                raise AssetImportError("ASSET_COMPANION_QUALIFICATION_FORBIDDEN")
            if inspected.asset_ir.source.adapter_id != adapter_id:
                raise AssetImportError("ASSET_COMPANION_PROVENANCE_ADAPTER_MISMATCH")
            if inspected.asset_ir.source.source_sha256 != source_package_sha256:
                raise AssetImportError("ASSET_COMPANION_PROVENANCE_HASH_MISMATCH")
            if (
                job.detected_source_format is not None
                and inspected.asset_ir.source.source_format != job.detected_source_format
            ):
                raise AssetImportError("ASSET_COMPANION_PROVENANCE_FORMAT_MISMATCH")
            job = self._bind_identity(job, inspected)
            destination = source.parent / "normalized.ddpkg"
            if destination.exists():
                raise AssetImportError("ASSET_COMPANION_RESULT_ALREADY_ATTACHED")
            temporary = source.parent / f".normalized-{uuid4().hex}.ddpkg"
            owns_temporary = False
            try:
                with temporary.open("xb") as output_file:
                    owns_temporary = True
                    copy_asset_source(result, output_file, MAX_PACKAGE_ARCHIVE_BYTES)
                copied = inspect_ddpkg(temporary)
                if copied.model_dump(mode="json") != inspected.model_dump(mode="json"):
                    raise AssetImportError("ASSET_COMPANION_RESULT_COPY_MISMATCH")
                check_plain_plugin_path(destination)
                os.link(temporary, destination)
            finally:
                if owns_temporary:
                    with suppress(OSError):
                        temporary.unlink()
            job = transition_import_job(
                job,
                "normalizing",
                progress_percent=max(40, job.progress_percent),
                asset_id=inspected.manifest.asset_id,
                asset_kind=inspected.manifest.asset_kind,
                normalized_content_sha256=inspected.manifest.content_sha256,
            )
            self.store.save_asset_import_job(job)
            record = self._process_locked(job_id)
            return record

    # 功能：
    #   1. 按持久状态推进检测、规范化、检查和发布，核对每一步内容身份。
    #   2. 未取得本地资格时保持等待；失败保存稳定错误代码，不执行包内原生代码。
    # 输入：
    #   self：已持有状态锁的导入服务。
    #   job_id：当前需要推进的任务标识。
    # 输出：
    #   record：当前阶段的持久任务记录。
    def _process_locked(self, job_id: str) -> dict[str, object]:
        job = self.store.load_asset_import_job(job_id)
        if job.state in _TERMINAL_STATES:
            raise AssetImportError(f"ASSET_IMPORT_JOB_TERMINAL:{job.state}")
        if job.state not in _PROCESSABLE_STATES:
            raise AssetImportError(f"ASSET_IMPORT_JOB_NOT_PROCESSABLE:{job.state}")
        try:
            if job.state == "quarantining":
                job = transition_import_job(job, "parsing", progress_percent=15)
                self.store.save_asset_import_job(job)
            source = self.store.get_asset_import_source(job_id)
            if self._sha256(source) != job.package_sha256:
                raise AssetImportError("ASSET_IMPORT_SOURCE_HASH_MISMATCH")
            normalized = source.parent / "normalized.ddpkg"
            if job.state == "needs_input":
                next_state = "normalizing" if normalized.is_file() else "parsing"
                job = transition_import_job(
                    job,
                    next_state,
                    progress_percent=max(40, job.progress_percent)
                    if next_state == "normalizing"
                    else job.progress_percent,
                )
                self.store.save_asset_import_job(job)
            if job.state == "parsing":
                detection = self.source_detector(source, job.source_format)
                job = self._bind_detection(job, detection)
                self.store.save_asset_import_job(job)
                if not detection.can_normalize_locally:
                    plugin_result = (
                        self.plugin_source_normalizer(
                            source=source,
                            detection=detection,
                            destination=normalized,
                            expected_kind=job.asset_kind,
                        )
                        if self.plugin_source_normalizer is not None
                        else None
                    )
                    if plugin_result is None:
                        required_input = f"plugin_adapter:{detection.adapter_id}"
                        record = self._wait_for_input_or_invalidate(
                            job, required_input=required_input
                        )
                        return record
                    normalized = plugin_result
                else:
                    normalized = normalize_asset_source(
                        source,
                        detection,
                        normalized,
                        expected_kind=job.asset_kind,
                    )
            elif not normalized.is_file():
                # 持久任务可能早于规范化文件落盘机制。恢复必须重新走当前准入，
                # 不能因为是旧任务就执行原生源文件或跳过伴随结果的来源绑定。
                detection = self.source_detector(source, job.source_format)
                job = self._bind_detection(job, detection)
                if not detection.can_normalize_locally:
                    plugin_result = (
                        self.plugin_source_normalizer(
                            source=source,
                            detection=detection,
                            destination=normalized,
                            expected_kind=job.asset_kind,
                        )
                        if self.plugin_source_normalizer is not None
                        else None
                    )
                    if plugin_result is None:
                        record = self._wait_for_input_or_invalidate(
                            job,
                            required_input=f"plugin_adapter:{detection.adapter_id}",
                        )
                        return record
                    normalized = plugin_result
                else:
                    normalized = normalize_asset_source(
                        source,
                        detection,
                        normalized,
                        expected_kind=job.asset_kind,
                    )
            inspected = inspect_ddpkg(normalized)
            job = self._bind_identity(job, inspected)
            if job.state == "parsing":
                job = transition_import_job(
                    job,
                    "normalizing",
                    progress_percent=max(40, job.progress_percent),
                    asset_id=inspected.manifest.asset_id,
                    asset_kind=inspected.manifest.asset_kind,
                    normalized_content_sha256=inspected.manifest.content_sha256,
                )
                self.store.save_asset_import_job(job)
            elif job.normalized_content_sha256 is None:
                job = AssetImportJob.model_validate(
                    {
                        **job.model_dump(mode="python"),
                        "normalized_content_sha256": inspected.manifest.content_sha256,
                    }
                )
                self.store.save_asset_import_job(job)
            elif job.normalized_content_sha256 != inspected.manifest.content_sha256:
                raise AssetImportError("ASSET_IMPORT_NORMALIZED_HASH_MISMATCH")
            qualification = inspected.manifest.qualification
            environment_versions = self.environment_versions_provider()
            if qualification is None or qualification.maturity != "qualified":
                self._admit_unqualified(normalized, inspected, job.job_id)
                record = self._wait_for_input_or_invalidate(
                    job, required_input="qualification_evidence"
                )
                return record
            if not environment_versions or not qualification_is_current(
                qualification,
                content_sha256=inspected.manifest.content_sha256,
                environment_versions=environment_versions,
            ):
                record = self._wait_for_input_or_invalidate(
                    job, required_input="qualification_environment_versions"
                )
                return record
            if self.local_qualification_verifier is None:
                record = self._wait_for_input_or_invalidate(
                    job, required_input="local_qualification_run"
                )
                return record
            try:
                locally_qualified = self.local_qualification_verifier(
                    inspected, environment_versions
                )
            except Exception as error:
                raise AssetImportError("LOCAL_QUALIFICATION_VERIFIER_FAILED") from error
            if locally_qualified is not True:
                record = self._wait_for_input_or_invalidate(
                    job, required_input="local_qualification_run"
                )
                return record
            if job.state == "normalizing":
                job = transition_import_job(
                    job, "building", progress_percent=max(65, job.progress_percent)
                )
                self.store.save_asset_import_job(job)
            if job.state == "building":
                job = transition_import_job(
                    job, "validating", progress_percent=max(90, job.progress_percent)
                )
                self.store.save_asset_import_job(job)
            if job.state == "validating":
                promoted = self._promote(normalized, inspected, job.job_id)
                self.store.record_asset_version(inspected, bundle_root=promoted)
                job = transition_import_job(
                    job,
                    "qualified",
                    progress_percent=100,
                    qualified_content_sha256=inspected.manifest.content_sha256,
                )
                record = self.store.save_asset_import_job(job)
                return record
            raise AssetImportError(f"ASSET_IMPORT_JOB_NOT_PROCESSABLE:{job.state}")
        except Exception as error:
            current = self.store.load_asset_import_job(job_id)
            if current.state in _TERMINAL_STATES:
                raise
            issue = self._public_issue_code(error)
            failed = transition_import_job(
                current,
                "failed",
                progress_percent=current.progress_percent,
                issue_codes=[issue],
            )
            self.store.save_asset_import_job(failed)
            raise AssetImportError(issue) from error

    # 功能：
    #   将可取消导入转为不可再次推进的取消状态，保留隔离材料和历史记录。
    # 输入：
    #   self：当前导入服务。
    #   job_id：待取消的任务标识。
    # 输出：
    #   record：已保存的取消记录。
    def cancel(self, job_id: str) -> dict[str, object]:
        with self._state_lock:
            job = self.store.load_asset_import_job(job_id)
            if job.state in _TERMINAL_STATES:
                raise AssetImportError(f"ASSET_IMPORT_JOB_TERMINAL:{job.state}")
            cancelled = transition_import_job(
                job,
                "cancelled",
                progress_percent=job.progress_percent,
            )
            record = self.store.save_asset_import_job(cancelled)
            return record

    # 功能：
    #   本地配对验收成功后，按原始内容摘要关闭对应等待中的导入任务。
    # 输入：
    #   self：当前导入服务。
    #   asset_id：验收对应的资产标识。
    #   source_content_sha256：被验收的原始内容摘要。
    #   result_content_sha256：验收后新版本的内容摘要。
    # 输出：
    #   completed：本次实际关闭的任务记录列表。
    def finalize_qualified_admissions(
        self,
        *,
        asset_id: str,
        source_content_sha256: str,
        result_content_sha256: str,
    ) -> list[dict[str, object]]:
        with self._state_lock:
            version = self.store.get_asset_version(asset_id, result_content_sha256)
            if version["maturity"] != "qualified":
                raise AssetImportError("ASSET_IMPORT_QUALIFIED_VERSION_REQUIRED")
            completed: list[dict[str, object]] = []
            for value in self.store.list_asset_import_jobs():
                job = AssetImportJob.model_validate(value)
                if (
                    job.state != "needs_input"
                    or job.asset_id != asset_id
                    or job.normalized_content_sha256 != source_content_sha256
                    or not set(job.required_inputs).intersection(_QUALIFICATION_REQUIRED_INPUTS)
                ):
                    continue
                expected_kind = str(version["kind"])
                if not self._asset_kinds_compatible(job.asset_kind, expected_kind):
                    raise AssetImportError("ASSET_IMPORT_KIND_MISMATCH")
                job = transition_import_job(
                    job,
                    "normalizing",
                    progress_percent=max(45, job.progress_percent),
                )
                self.store.save_asset_import_job(job)
                job = transition_import_job(
                    job, "building", progress_percent=max(70, job.progress_percent)
                )
                self.store.save_asset_import_job(job)
                job = transition_import_job(
                    job, "validating", progress_percent=max(90, job.progress_percent)
                )
                self.store.save_asset_import_job(job)
                job = transition_import_job(
                    job,
                    "qualified",
                    progress_percent=100,
                    qualified_content_sha256=result_content_sha256,
                )
                completed.append(self.store.save_asset_import_job(job))
            return completed

    # 功能：
    #   按资产及内容摘要选择已安装版本，用共享的有界 I/O 重建无覆盖导出包。
    # 输入：
    #   self：当前导入服务。
    #   asset_id：选定资产标识。
    #   content_sha256：选定不可变内容摘要。
    #   destination：尚不存在的导出路径。
    # 输出：
    #   exported：通过完整索引检查的新包路径。
    def export_version(
        self,
        *,
        asset_id: str,
        content_sha256: str,
        destination: Path,
    ) -> Path:
        version = self.store.get_asset_version(asset_id, content_sha256)
        root = Path(str(version["bundle_root"]))
        check_plain_plugin_path(root)
        root = root.resolve()
        versions_root = self.store.asset_versions_root.resolve()
        if versions_root not in root.parents or not root.is_dir():
            raise AssetImportError("ASSET_VERSION_PATH_INVALID")
        try:
            exported = export_stored_asset(
                root,
                destination,
                expected_asset_id=asset_id,
                expected_content_sha256=content_sha256,
            )
        except (ValueError, OSError) as error:
            raise AssetImportError(self._public_issue_code(error)) from error
        return exported

    # 功能：
    #   安装当前发布绑定的默认配对，随后按精确摘要迁移旧记录，不恢复退役 ZIP 流程。
    # 输入：
    #   self：当前导入服务。
    #   bundle_directory：产品随附默认资产目录。
    # 输出：
    #   rows：安装或复用后的默认资产记录列表。
    def seed_bundled_sources(self, bundle_directory: Path) -> list[dict[str, object]]:
        index_path = bundle_directory / "index.json"
        if not index_path.is_file():
            rows: list[dict[str, object]] = []
            return rows
        try:
            index = decode_json(
                read_plugin_file(index_path, limit=MAX_MANIFEST_BYTES), limit=MAX_MANIFEST_BYTES
            )
        except (OSError, ValueError) as error:
            raise AssetImportError("BUNDLED_ASSET_INDEX_INVALID") from error
        if not isinstance(index, dict):
            raise AssetImportError("BUNDLED_ASSET_INDEX_INVALID")
        schema_version = index.get("schema_version")
        if schema_version not in {
            "dronedream.bundled-assets.v2",
            "dronedream.bundled-assets.v3",
        }:
            raise AssetImportError("BUNDLED_ASSET_INDEX_INVALID")
        current_pair = index.get("qualified_pair")
        if not isinstance(current_pair, dict):
            raise AssetImportError("BUNDLED_QUALIFIED_PAIR_INVALID")
        if schema_version == "dronedream.bundled-assets.v2":
            qualified_pairs = [current_pair]
        else:
            qualified_pairs = index.get("qualified_pairs")
            default_qualification_id = index.get("default_qualification_id")
            if (
                not isinstance(qualified_pairs, list)
                or not 1 <= len(qualified_pairs) <= 64
                or not isinstance(default_qualification_id, str)
                or current_pair.get("qualification_id") != default_qualification_id
            ):
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_INVALID")
            qualification_ids: list[str] = []
            resource_ids: list[str] = []
            for pair in qualified_pairs:
                if not isinstance(pair, dict):
                    raise AssetImportError("BUNDLED_QUALIFIED_PAIR_INVALID")
                qualification_id = pair.get("qualification_id")
                resource_id = pair.get("resource_id")
                if not isinstance(qualification_id, str) or not isinstance(resource_id, str):
                    raise AssetImportError("BUNDLED_QUALIFIED_PAIR_INVALID")
                qualification_ids.append(qualification_id)
                resource_ids.append(resource_id)
            if (
                len(qualification_ids) != len(set(qualification_ids))
                or len(resource_ids) != len(set(resource_ids))
                or default_qualification_id not in qualification_ids
                or not any(pair == current_pair for pair in qualified_pairs)
            ):
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_INVALID")
        # 历史迁移声明也必须在任何安装/入库之前验证，不能先安装再发现索引损坏。
        superseded_pairs = index.get("superseded_qualified_pairs", [])
        if not isinstance(superseded_pairs, list):
            raise AssetImportError("BUNDLED_SUPERSEDED_PAIR_INVALID")
        for superseded in superseded_pairs:
            if not isinstance(superseded, dict):
                raise AssetImportError("BUNDLED_SUPERSEDED_PAIR_INVALID")
            qualification_id = superseded.get("qualification_id")
            if (
                not isinstance(qualification_id, str)
                or re.fullmatch(r"asset-qualification-[0-9a-f]{24}", qualification_id) is None
            ):
                raise AssetImportError("BUNDLED_SUPERSEDED_PAIR_INVALID")
            for field in (
                "map_content_sha256",
                "map_source_content_sha256",
                "vehicle_content_sha256",
                "vehicle_source_content_sha256",
            ):
                digest = superseded.get(field)
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise AssetImportError("BUNDLED_SUPERSEDED_PAIR_INVALID")
            if isinstance(current_pair, dict) and qualification_id == current_pair.get(
                "qualification_id"
            ):
                raise AssetImportError("BUNDLED_SUPERSEDED_PAIR_IS_CURRENT")
        rows: list[dict[str, object]] = []
        for pair in qualified_pairs:
            rows.extend(
                self._seed_bundled_qualified_pair(
                    bundle_directory=bundle_directory,
                    pair=pair,
                )
            )
        packages = current_pair.get("packages")
        if not isinstance(packages, list):
            raise AssetImportError("BUNDLED_QUALIFIED_PAIR_INVALID")
        replacements = {
            str(package.get("kind")): package for package in packages if isinstance(package, dict)
        }
        for superseded in superseded_pairs:
            qualification_id = superseded.get("qualification_id")
            map_sha256 = superseded.get("map_content_sha256")
            map_source_sha256 = superseded.get("map_source_content_sha256")
            vehicle_sha256 = superseded.get("vehicle_content_sha256")
            vehicle_source_sha256 = superseded.get("vehicle_source_content_sha256")
            self.store.retire_bundled_asset_pair_qualification(
                qualification_id=qualification_id,
                map_content_sha256=map_sha256,
                vehicle_content_sha256=vehicle_sha256,
            )
            for kind, source_sha256 in (
                ("map", map_sha256),
                ("map", map_source_sha256),
                ("vehicle", vehicle_sha256),
                ("vehicle", vehicle_source_sha256),
            ):
                replacement = replacements.get(kind)
                if not isinstance(replacement, dict):
                    raise AssetImportError("BUNDLED_QUALIFIED_PAIR_ENTRY_INVALID")
                self.store.migrate_bundled_asset_version(
                    asset_id=str(replacement["asset_id"]),
                    source_content_sha256=source_sha256,
                    replacement_content_sha256=str(replacement["content_sha256"]),
                )
        for package in packages:
            if not isinstance(package, dict):
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_ENTRY_INVALID")
            self.store.migrate_bundled_asset_version(
                asset_id=str(package["asset_id"]),
                source_content_sha256=str(package["source_content_sha256"]),
                replacement_content_sha256=str(package["content_sha256"]),
            )
        return rows

    # 功能：
    #   1. 核对发布索引、两个包及共同回执，安装声明的精确默认配对。
    #   2. 保存可恢复的配对记录，再核查实际入库材料与声明摘要相同。
    # 输入：
    #   self：当前导入服务。
    #   bundle_directory：默认包所在目录。
    #   index：已完成有界解析的当前默认资产索引。
    # 输出：
    #   rows：两个已安装默认资产的记录。
    def _seed_bundled_qualified_pair(
        self,
        *,
        bundle_directory: Path,
        pair: dict[str, object],
    ) -> list[dict[str, object]]:
        if pair.get("schema_version") != (
            "dronedream.bundled-qualified-pair.v1"
        ):
            raise AssetImportError("BUNDLED_QUALIFIED_PAIR_INVALID")
        packages = pair.get("packages")
        qualification_id = pair.get("qualification_id")
        receipt_sha256 = pair.get("receipt_sha256")
        if (
            not isinstance(packages, list)
            or len(packages) != 2
            or not isinstance(qualification_id, str)
            or not isinstance(receipt_sha256, str)
        ):
            raise AssetImportError("BUNDLED_QUALIFIED_PAIR_INVALID")

        check_plain_plugin_path(bundle_directory)
        root = bundle_directory.resolve()
        inspected_by_kind: dict[str, tuple[Path, InspectedDDPkg, dict[str, object]]] = {}
        expected_evidence_path = f"evidence/qualification/{qualification_id}.json"
        for raw_entry in packages:
            if not isinstance(raw_entry, dict) or raw_entry.get("kind") not in {
                "map",
                "vehicle",
            }:
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_ENTRY_INVALID")
            required = (
                raw_entry.get("file"),
                raw_entry.get("sha256"),
                raw_entry.get("asset_id"),
                raw_entry.get("content_sha256"),
                raw_entry.get("source_content_sha256"),
            )
            if not all(isinstance(value, str) and value for value in required):
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_ENTRY_INVALID")
            kind = str(raw_entry["kind"])
            if kind in inspected_by_kind:
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_ENTRY_DUPLICATE")
            try:
                relative = normalized_member_path(raw_entry["file"])
                if relative != raw_entry["file"]:
                    raise ValueError("noncanonical package path")
                archive = bundle_directory / relative
                check_plain_plugin_path(archive)
                archive = archive.resolve()
            except (OSError, ValueError) as error:
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_FILE_INVALID") from error
            if root not in archive.parents or not archive.is_file():
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_FILE_INVALID")
            if self._sha256(archive) != raw_entry["sha256"]:
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_HASH_MISMATCH")
            inspected = inspect_ddpkg(archive)
            qualification = inspected.manifest.qualification
            if (
                inspected.manifest.asset_id != raw_entry["asset_id"]
                or inspected.manifest.asset_kind != kind
                or inspected.manifest.content_sha256 != raw_entry["content_sha256"]
                or qualification is None
                or qualification.maturity != "qualified"
                or qualification.content_sha256 != raw_entry["content_sha256"]
                or qualification.evidence_paths != [expected_evidence_path]
            ):
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_BINDING_INVALID")
            try:
                evidence = next(
                    entry
                    for entry in inspected.manifest.files
                    if entry.path == expected_evidence_path
                )
            except StopIteration as error:
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_EVIDENCE_MISSING") from error
            if evidence.sha256 != receipt_sha256:
                raise AssetImportError("BUNDLED_QUALIFIED_PAIR_RECEIPT_HASH_MISMATCH")
            inspected_by_kind[kind] = (archive, inspected, raw_entry)

        if set(inspected_by_kind) != {"map", "vehicle"}:
            raise AssetImportError("BUNDLED_QUALIFIED_PAIR_INCOMPLETE")
        map_archive, map_inspected, map_entry = inspected_by_kind["map"]
        vehicle_archive, vehicle_inspected, vehicle_entry = inspected_by_kind["vehicle"]
        map_qualification = map_inspected.manifest.qualification
        vehicle_qualification = vehicle_inspected.manifest.qualification
        if (
            map_qualification is None
            or vehicle_qualification is None
            or map_qualification.model_dump(exclude={"content_sha256"})
            != vehicle_qualification.model_dump(exclude={"content_sha256"})
        ):
            raise AssetImportError("BUNDLED_QUALIFIED_PAIR_BINDING_MISMATCH")

        receipt = self._verify_pair_receipts(
            ((map_archive, map_inspected), (vehicle_archive, vehicle_inspected)),
            map_qualification.environment_versions,
        )
        if (
            receipt.qualification_id != qualification_id
            or receipt.map_content_sha256 != map_entry["source_content_sha256"]
            or receipt.vehicle_content_sha256 != vehicle_entry["source_content_sha256"]
        ):
            raise AssetImportError("BUNDLED_QUALIFIED_PAIR_BINDING_MISMATCH")

        operation_id = f"bundled-pair-{receipt_sha256[:16]}"
        map_root = self._promote(map_archive, map_inspected, f"{operation_id}-map")
        vehicle_root = self._promote(
            vehicle_archive,
            vehicle_inspected,
            f"{operation_id}-vehicle",
        )
        rows = self.store.record_asset_versions(
            [(map_inspected, map_root), (vehicle_inspected, vehicle_root)]
        )
        qualified_at = map_qualification.qualified_at
        job = AssetPairQualificationJob(
            job_id=(
                "asset-qualification-job-"
                + hashlib.sha256(f"{qualification_id}:{receipt_sha256}".encode()).hexdigest()[:24]
            ),
            owner_id="bundled-release",
            map_asset_id=str(map_entry["asset_id"]),
            map_content_sha256=str(map_entry["source_content_sha256"]),
            vehicle_asset_id=str(vehicle_entry["asset_id"]),
            vehicle_content_sha256=str(vehicle_entry["source_content_sha256"]),
            state="qualified",
            progress_percent=100,
            revision=1,
            qualification_id=qualification_id,
            result_map_content_sha256=map_inspected.manifest.content_sha256,
            result_vehicle_content_sha256=vehicle_inspected.manifest.content_sha256,
            created_at=qualified_at,
            updated_at=qualified_at,
        )
        workspace = self.store.asset_qualification_root / f"bundled-{qualification_id}"
        workspace.mkdir(exist_ok=True)
        self.store.save_asset_pair_qualification_job(
            job,
            workspace_root=workspace,
            map_bundle_root=map_root,
            vehicle_bundle_root=vehicle_root,
        )
        _receipt, verified_sha256 = self.store.verified_asset_pair_receipt(job)
        if verified_sha256 != receipt_sha256:
            raise AssetImportError("BUNDLED_QUALIFIED_PAIR_RECEIPT_HASH_MISMATCH")
        return rows

    # 功能：
    #   安装本地验收路径产生的单个包，要求资产身份和实际环境绑定一致。
    # 输入：
    #   self：当前导入服务。
    #   source：本地验收输出包。
    #   expected_asset_id：本次验收的资产标识。
    #   environment_versions：实际验收环境标识。
    #   operation_id：本次安装操作标识。
    # 输出：
    #   record：入库后的资产版本记录。
    def promote_locally_verified_package(
        self,
        source: Path,
        *,
        expected_asset_id: str,
        environment_versions: dict[str, str],
        operation_id: str,
    ) -> dict[str, object]:
        inspected = inspect_ddpkg(source)
        if inspected.manifest.asset_id != expected_asset_id:
            raise AssetImportError("ASSET_QUALIFICATION_RESULT_IDENTITY_MISMATCH")
        qualification = inspected.manifest.qualification
        if not environment_versions or not qualification_is_current(
            qualification,
            content_sha256=inspected.manifest.content_sha256,
            environment_versions=environment_versions,
        ):
            raise AssetImportError("ASSET_QUALIFICATION_RESULT_BINDING_INVALID")
        if qualification is None or qualification.maturity != "qualified":
            raise AssetImportError("ASSET_QUALIFICATION_RESULT_NOT_QUALIFIED")
        promoted = self._promote(source, inspected, operation_id)
        record = self.store.record_asset_version(inspected, bundle_root=promoted)
        return record

    # 功能：
    #   1. 核对本地配对结果后，同时提交两条资产版本数据库记录。
    #   2. 失败时保留已发布的不可变内容，不删除可能被另一流程复用的目录。
    # 输入：
    #   self：当前导入服务。
    #   map_source：本地验收后的地图包。
    #   map_asset_id：预期地图标识。
    #   vehicle_source：本地验收后的飞机包。
    #   vehicle_asset_id：预期飞机标识。
    #   environment_versions：配对验收使用的运行环境。
    #   operation_id：本次安装操作标识。
    # 输出：
    #   pair：同时提交的地图记录与飞机记录组成的元组。
    def promote_locally_verified_pair(
        self,
        *,
        map_source: Path,
        map_asset_id: str,
        vehicle_source: Path,
        vehicle_asset_id: str,
        environment_versions: dict[str, str],
        operation_id: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        inputs = ((map_source, map_asset_id), (vehicle_source, vehicle_asset_id))
        inspected_values: list[InspectedDDPkg] = []
        for source, expected_asset_id in inputs:
            inspected = inspect_ddpkg(source)
            if inspected.manifest.asset_id != expected_asset_id:
                raise AssetImportError("ASSET_QUALIFICATION_RESULT_IDENTITY_MISMATCH")
            qualification = inspected.manifest.qualification
            if qualification is None or qualification.maturity != "qualified":
                raise AssetImportError("ASSET_QUALIFICATION_RESULT_NOT_QUALIFIED")
            if not environment_versions or not qualification_is_current(
                qualification,
                content_sha256=inspected.manifest.content_sha256,
                environment_versions=environment_versions,
            ):
                raise AssetImportError("ASSET_QUALIFICATION_RESULT_BINDING_INVALID")
            inspected_values.append(inspected)

        self._verify_pair_receipts(
            tuple(
                (source, inspected)
                for (source, _), inspected in zip(inputs, inspected_values, strict=True)
            ),
            environment_versions,
        )
        promoted: list[Path] = []
        for index, ((source, _asset_id), inspected) in enumerate(
            zip(inputs, inspected_values, strict=True)
        ):
            promoted.append(self._promote(source, inspected, f"{operation_id}-{index}"))
        # 内容目录没有数据库可见性；只有两份内容都准备好才原子提交记录。
        # 提交失败保留可恢复内容，不能用“开始时不存在”推断当前目录的所有权。
        rows = self.store.record_asset_versions(list(zip(inspected_values, promoted, strict=True)))
        pair = rows[0], rows[1]
        return pair

    # 功能：
    #   1. 在任何发布或数据库写入前，确认地图与飞机包含同一份有效配对回执。
    #   2. 核对回执实际字节、双方身份和环境，拒绝把两个独立合格包拼成假配对。
    # 输入：
    #   packages：按地图、飞机顺序排列的源包路径和检查结果。
    #   environment：调用方确认的验收环境。
    # 输出：
    #   receipt：双方共同携带的已验证配对回执。
    @staticmethod
    def _verify_pair_receipts(
        packages: tuple[tuple[Path, InspectedDDPkg], ...],
        environment: dict[str, str],
    ) -> AssetPairQualificationReceipt:
        if (
            len(packages) != 2
            or packages[0][1].manifest.asset_kind not in {"map", "world"}
            or packages[1][1].manifest.asset_kind != "vehicle"
        ):
            raise AssetImportError("ASSET_QUALIFICATION_RESULT_KIND_MISMATCH")
        receipts: list[AssetPairQualificationReceipt] = []
        payload_hashes: list[str] = []
        for path, inspected in packages:
            binding = inspected.manifest.qualification
            if binding is None or len(binding.evidence_paths) != 1:
                raise AssetImportError("ASSET_QUALIFICATION_PAIR_EVIDENCE_INVALID")
            evidence_path = binding.evidence_paths[0]
            entry = next(
                (item for item in inspected.manifest.files if item.path == evidence_path), None
            )
            if entry is None or entry.size_bytes > MAX_ASSET_IR_BYTES:
                raise AssetImportError("ASSET_QUALIFICATION_PAIR_EVIDENCE_INVALID")
            with open_verified_ddpkg(path) as (archive, current):
                if current.model_dump(mode="json") != inspected.model_dump(mode="json"):
                    raise AssetImportError("ASSET_QUALIFICATION_PAIR_SOURCE_CHANGED")
                with archive.open(evidence_path) as source:
                    payload = source.read(MAX_ASSET_IR_BYTES + 1)
            if (
                len(payload) != entry.size_bytes
                or hashlib.sha256(payload).hexdigest() != entry.sha256
            ):
                raise AssetImportError("ASSET_QUALIFICATION_PAIR_EVIDENCE_CHANGED")
            try:
                receipt = AssetPairQualificationReceipt.model_validate(
                    decode_json(payload, limit=MAX_ASSET_IR_BYTES, node_limit=2_000_000)
                )
            except ValueError as error:
                raise AssetImportError("ASSET_QUALIFICATION_PAIR_EVIDENCE_INVALID") from error
            if (
                evidence_path != f"evidence/qualification/{receipt.qualification_id}.json"
                or receipt.map_asset_id != packages[0][1].manifest.asset_id
                or receipt.vehicle_asset_id != packages[1][1].manifest.asset_id
                or receipt.environment_versions != environment
                or binding.environment_versions != environment
                or receipt.qualified_at != binding.qualified_at
            ):
                raise AssetImportError("ASSET_QUALIFICATION_PAIR_BINDING_MISMATCH")
            receipts.append(receipt)
            payload_hashes.append(entry.sha256)
        if payload_hashes[0] != payload_hashes[1]:
            raise AssetImportError("ASSET_QUALIFICATION_PAIR_RECEIPT_MISMATCH")
        receipt = receipts[0]
        return receipt

    # 功能：
    #   将允许补资料的任务转为等待；较晚阶段资格失效时拒绝隐式回退。
    # 输入：
    #   self：当前导入服务。
    #   job：当前任务契约。
    #   required_input：继续处理所缺的材料或本地能力。
    # 输出：
    #   record：已保存的等待记录。
    def _wait_for_input_or_invalidate(
        self,
        job: AssetImportJob,
        *,
        required_input: str,
    ) -> dict[str, object]:
        if job.state not in {"parsing", "normalizing"}:
            raise AssetImportError("ASSET_IMPORT_QUALIFICATION_INVALIDATED")
        waiting = transition_import_job(
            job,
            "needs_input",
            progress_percent=max(25, job.progress_percent),
            required_inputs=[required_input],
        )
        record = self.store.save_asset_import_job(waiting)
        return record

    # 功能：
    #   保存可查看的规范化资产，不提升其声明成熟度，也不授予飞行资格。
    # 输入：
    #   self：当前导入服务。
    #   source：已检查的规范化包。
    #   inspected：该包的检查结果。
    #   operation_id：导入操作标识。
    # 输出：
    #   record：保留原成熟度的版本记录。
    def _admit_unqualified(
        self,
        source: Path,
        inspected: InspectedDDPkg,
        operation_id: str,
    ) -> dict[str, object]:
        promoted = self._promote(source, inspected, f"{operation_id}-admission")
        record = self.store.record_asset_version(inspected, bundle_root=promoted)
        return record

    # 功能：
    #   将异常转换为有限长度的产品错误代码，不把任意异常原文存进任务记录。
    # 输入：
    #   error：本次捕获的异常。
    # 输出：
    #   code：可对外显示的稳定错误代码。
    @staticmethod
    def _public_issue_code(error: Exception) -> str:
        candidate = str(error).split(":", 1)[0]
        if re.fullmatch(r"(?:ASSET|DDPKG|BUNDLED|LOCAL|PLUGIN_FILE)_[A-Z0-9_]{1,80}", candidate):
            code = candidate
        elif isinstance(error, zipfile.BadZipFile):
            code = "DDPKG_NOT_ZIP"
        elif isinstance(error, OSError):
            code = "ASSET_IMPORT_IO_FAILURE"
        else:
            code = "ASSET_IMPORT_INTERNAL"
        return code

    # 功能：
    #   将检测到的资产身份绑定到原任务，拒绝后续更换标识或不兼容类别。
    # 输入：
    #   job：当前导入任务。
    #   inspected：当前包检查结果。
    # 输出：
    #   updated_job：重新验证身份字段后的任务副本。
    @staticmethod
    def _bind_identity(job: AssetImportJob, inspected: InspectedDDPkg) -> AssetImportJob:
        if job.asset_id is not None and job.asset_id != inspected.manifest.asset_id:
            raise AssetImportError("ASSET_IMPORT_IDENTITY_MISMATCH")
        if job.asset_kind is not None and not AssetImportService._asset_kinds_compatible(
            job.asset_kind, inspected.manifest.asset_kind
        ):
            raise AssetImportError("ASSET_IMPORT_KIND_MISMATCH")
        value = {
            **job.model_dump(mode="python"),
            "asset_id": inspected.manifest.asset_id,
            "asset_kind": inspected.manifest.asset_kind,
        }
        updated_job = AssetImportJob.model_validate(value)
        return updated_job

    # 功能：
    #   判断资产类别是否一致；未限定类别或地图与世界的兼容组合可以匹配。
    # 输入：
    #   left：原先限定的类别。
    #   right：本次检测的类别。
    # 输出：
    #   compatible：两个类别是否兼容。
    @staticmethod
    def _asset_kinds_compatible(left: str | None, right: str | None) -> bool:
        compatible = (
            left is None or right is None or left == right or {left, right} == {"map", "world"}
        )
        return compatible

    # 功能：
    #   绑定原始格式及适配器，防止任务恢复时切换到与原检测不一致的转换链路。
    # 输入：
    #   job：当前导入任务。
    #   detection：本次原始格式检测结果。
    # 输出：
    #   updated_job：经过重新校验的检测绑定副本。
    @staticmethod
    def _bind_detection(job: AssetImportJob, detection: AssetSourceDetection) -> AssetImportJob:
        if (
            job.detected_source_format is not None
            and job.detected_source_format != detection.source_format
        ):
            raise AssetImportError("ASSET_IMPORT_SOURCE_DETECTION_CHANGED")
        if job.source_adapter_id is not None and job.source_adapter_id != detection.adapter_id:
            raise AssetImportError("ASSET_IMPORT_SOURCE_ADAPTER_CHANGED")
        if (
            job.asset_kind is not None
            and detection.asset_kind is not None
            and not AssetImportService._asset_kinds_compatible(job.asset_kind, detection.asset_kind)
        ):
            raise AssetImportError("ASSET_IMPORT_KIND_MISMATCH")
        value = {
            **job.model_dump(mode="python"),
            "detected_source_format": detection.source_format,
            "source_adapter_id": detection.adapter_id,
            "asset_kind": job.asset_kind if job.asset_kind is not None else detection.asset_kind,
        }
        updated_job = AssetImportJob.model_validate(value)
        return updated_job

    # 功能：
    #   在共享状态锁下发布内容版本，复用已有目录之前完整复核其实际内容。
    # 输入：
    #   self：当前导入服务。
    #   source：经过隔离的包路径。
    #   inspected：源包检查结果。
    #   job_id：关联操作标识。
    # 输出：
    #   destination：已验证的不可变版本目录。
    def _promote(self, source: Path, inspected: InspectedDDPkg, job_id: str) -> Path:
        with self._state_lock:
            destination = self._promote_locked(source, inspected, job_id)
            return destination

    # 功能：
    #   1. 提取到独占暂存目录，复核全部索引后无覆盖发布。
    #   2. 并发者先发布时仅复核并复用其相同内容；只清理自己仍持有的暂存目录。
    # 输入：
    #   self：已持有状态锁的导入服务。
    #   source：源资产包。
    #   inspected：先前检查结果。
    #   job_id：用于生成不含路径含义的暂存名称的操作标识。
    # 输出：
    #   destination：完成验证的最终内容目录。
    def _promote_locked(self, source: Path, inspected: InspectedDDPkg, job_id: str) -> Path:
        inspected = InspectedDDPkg.model_validate(inspected.model_dump(mode="python"))
        version_root = self.store.asset_versions_root.resolve()
        destination = self._version_destination(inspected)
        if version_root not in destination.parents:
            raise AssetImportError("ASSET_VERSION_PATH_INVALID")
        if destination.is_dir():
            self._verify_promoted_files(destination, inspected)
            return destination
        operation_key = hashlib.sha256(job_id.encode()).hexdigest()[:16]
        staging = version_root / f".staging-{operation_key}-{uuid4().hex}"
        if version_root not in staging.parents:
            raise AssetImportError("ASSET_VERSION_STAGING_PATH_INVALID")
        staging.mkdir(parents=False)
        owned_directory = staging.stat()
        try:
            extract_verified_asset(source, staging, inspected)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                publish_asset_directory(staging, destination)
            except FileExistsError:
                self._verify_promoted_files(destination, inspected)
            return destination
        finally:
            if staging.is_dir():
                check_plain_plugin_path(staging)
                if version_root in staging.resolve().parents and os.path.samestat(
                    owned_directory, staging.stat()
                ):
                    shutil.rmtree(staging)

    # 功能：
    #   根据包类别、资产标识和内容摘要确定唯一安装位置，不按时间选择旧目录。
    # 输入：
    #   self：当前导入服务。
    #   inspected：已验证的包元数据。
    # 输出：
    #   destination：限定在资产版本根目录内的目标目录。
    def _version_destination(self, inspected: InspectedDDPkg) -> Path:
        version_root = self.store.asset_versions_root.resolve()
        destination = (
            version_root
            / inspected.manifest.asset_kind
            / inspected.manifest.asset_id
            / inspected.manifest.content_sha256
        )
        check_plain_plugin_path(destination)
        destination = destination.resolve()
        if version_root not in destination.parents:
            raise AssetImportError("ASSET_VERSION_PATH_INVALID")
        return destination

    # 功能：
    #   通过共享目录检查核对清单、成员摘要及额外文件，转换为导入接口错误。
    # 输入：
    #   self：当前导入服务。
    #   root：已安装或暂存目录。
    #   inspected：预期包元数据。
    # 输出：
    #   None：不返回业务数据。
    def _verify_promoted_files(self, root: Path, inspected: InspectedDDPkg) -> None:
        try:
            verify_stored_asset(root, inspected.manifest)
        except (ValueError, OSError) as error:
            raise AssetImportError(self._public_issue_code(error)) from error

    # 功能：
    #   有界读取实际源文件并核对读取期间的文件身份，计算稳定内容摘要。
    # 输入：
    #   path：普通文件路径。
    # 输出：
    #   digest：实际文件字节的 SHA-256。
    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hash_plugin_file(path, limit=MAX_PACKAGE_ARCHIVE_BYTES)
        return digest
