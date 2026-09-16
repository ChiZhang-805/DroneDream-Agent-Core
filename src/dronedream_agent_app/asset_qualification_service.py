"""Durable orchestration for real map-and-vehicle qualification runs."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_hex

from dronedream_agent_core.asset_package_storage import export_stored_asset
from dronedream_agent_core.asset_packages import (
    MAX_ASSET_IR_BYTES,
    AssetIR,
    AssetPackageError,
    inspect_ddpkg,
)
from dronedream_agent_core.asset_pair_qualification import (
    AssetPairQualificationError,
    AssetPairQualificationJob,
    bind_qualification_receipt,
    build_pair_qualification_receipt,
    prepare_asset_pair_qualification,
    transition_pair_qualification_job,
)
from dronedream_agent_core.contracts import MapAsset, VehicleAsset
from dronedream_agent_core.plugin_files import read_plugin_file
from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .asset_import_service import AssetImportService
from .asset_runtime_resolver import resolve_versioned_map, resolve_versioned_vehicle
from .storage import AppStore, AssetImportError

RuntimeRunner = Callable[..., dict[str, object]]
RuntimeCanceller = Callable[[str], bool]
EnvironmentProvider = Callable[[], dict[str, str]]
QualificationChecker = Callable[..., dict[str, object]]
_TERMINAL_STATES = {"qualified", "failed", "cancelled"}
_INTERRUPTED_STATES = {"preparing", "running", "validating"}


class AssetQualificationServiceError(ValueError):
    """A durable asset-pair qualification operation was rejected."""


class AssetQualificationService:
    """Prepare, execute, validate and atomically promote one exact asset pair."""

    # 功能：
    #   连接持久化存储、运行执行器和附加检查器，并恢复上次中断的任务状态。
    # 输入：
    #   self：当前验收服务。
    #   store：任务与资产数据库。
    #   asset_import_service：负责发布已验证资产的导入服务。
    #   runtime_runner：执行实际配对验收并返回证据的回调。
    #   runtime_canceller：请求取消指定运行的回调。
    #   environment_versions_provider：读取当前运行环境身份的回调。
    #   qualification_checker：可选的附加插件检查回调。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        store: AppStore,
        asset_import_service: AssetImportService,
        *,
        runtime_runner: RuntimeRunner,
        runtime_canceller: RuntimeCanceller,
        environment_versions_provider: EnvironmentProvider,
        qualification_checker: QualificationChecker | None = None,
    ) -> None:
        self.store = store
        self.asset_import_service = asset_import_service
        self.runtime_runner = runtime_runner
        self.runtime_canceller = runtime_canceller
        self.environment_versions_provider = environment_versions_provider
        self.qualification_checker = qualification_checker
        self._lock = threading.RLock()
        self._workers: dict[str, threading.Thread] = {}
        self._pause_interrupted_jobs()
        self._reconcile_qualified_import_jobs()

    # 功能：
    #   将进程退出前尚未完成的任务改为暂停，禁止把没有执行器的任务显示为运行中。
    # 输入：
    #   self：持有持久化任务存储的服务。
    # 输出：
    #   None：不返回业务数据。
    def _pause_interrupted_jobs(self) -> None:
        for value in self.store.list_asset_pair_qualification_jobs():
            job = AssetPairQualificationJob.model_validate(value)
            if job.state in _INTERRUPTED_STATES:
                paused = transition_pair_qualification_job(
                    job,
                    "paused",
                    progress_percent=job.progress_percent,
                )
                self.store.save_asset_pair_qualification_job(paused)

    # 功能：
    #   重启后补齐已合格配对对应的导入状态，不重新执行或伪造飞行验收。
    # 输入：
    #   self：持有验收记录和导入服务的当前服务。
    # 输出：
    #   None：不返回业务数据。
    def _reconcile_qualified_import_jobs(self) -> None:
        for value in self.store.list_asset_pair_qualification_jobs():
            job = AssetPairQualificationJob.model_validate(value)
            if (
                job.state != "qualified"
                or job.result_map_content_sha256 is None
                or job.result_vehicle_content_sha256 is None
            ):
                continue
            self.asset_import_service.finalize_qualified_admissions(
                asset_id=job.map_asset_id,
                source_content_sha256=job.map_content_sha256,
                result_content_sha256=job.result_map_content_sha256,
            )
            self.asset_import_service.finalize_qualified_admissions(
                asset_id=job.vehicle_asset_id,
                source_content_sha256=job.vehicle_content_sha256,
                result_content_sha256=job.result_vehicle_content_sha256,
            )

    # 功能：
    #   检查资产种类，为指定内容版本建立独立验收任务和工作目录。
    # 输入：
    #   self：当前验收服务。
    #   map_asset_id：地图或世界资产标识。
    #   map_content_sha256：待验收地图内容摘要。
    #   vehicle_asset_id：飞机资产标识。
    #   vehicle_content_sha256：待验收飞机内容摘要。
    #   owner_id：任务所有者标识。
    # 输出：
    #   record：保存后的新任务记录。
    def create(
        self,
        *,
        map_asset_id: str,
        map_content_sha256: str,
        vehicle_asset_id: str,
        vehicle_content_sha256: str,
        owner_id: str = "desktop-local-user",
    ) -> dict[str, object]:
        map_version = self.store.get_asset_version(map_asset_id, map_content_sha256)
        vehicle_version = self.store.get_asset_version(vehicle_asset_id, vehicle_content_sha256)
        if map_version["kind"] not in {"map", "world"}:
            raise AssetQualificationServiceError("ASSET_QUALIFICATION_MAP_KIND_REQUIRED")
        if vehicle_version["kind"] != "vehicle":
            raise AssetQualificationServiceError("ASSET_QUALIFICATION_VEHICLE_KIND_REQUIRED")
        now = datetime.now(UTC)
        job = AssetPairQualificationJob(
            job_id=f"asset-qualification-job-{token_hex(12)}",
            owner_id=owner_id,
            map_asset_id=map_asset_id,
            map_content_sha256=map_content_sha256,
            vehicle_asset_id=vehicle_asset_id,
            vehicle_content_sha256=vehicle_content_sha256,
            created_at=now,
            updated_at=now,
        )
        workspace = self.store.asset_qualification_root / job.job_id
        workspace.mkdir(parents=False)
        try:
            record = self.store.save_asset_pair_qualification_job(
                job,
                workspace_root=workspace,
                map_bundle_root=Path(str(map_version["bundle_root"])),
                vehicle_bundle_root=Path(str(vehicle_version["bundle_root"])),
            )
        except BaseException:
            workspace.rmdir()
            raise
        return record

    # 功能：
    #   1. 只启动新建或暂停的任务，阻止同一任务重复启动。
    #   2. 线程启动失败时保留可恢复的暂停状态。
    # 输入：
    #   self：当前验收服务。
    #   job_id：需要启动或恢复的任务标识。
    # 输出：
    #   snapshot：进入准备阶段时保存的任务记录。
    def start(self, job_id: str) -> dict[str, object]:
        with self._lock:
            existing_worker = self._workers.get(job_id)
            if existing_worker is not None and existing_worker.is_alive():
                raise AssetQualificationServiceError("ASSET_QUALIFICATION_ALREADY_RUNNING")
            job = self.store.load_asset_pair_qualification_job(job_id)
            if job.state not in {"created", "paused"}:
                raise AssetQualificationServiceError(
                    f"ASSET_QUALIFICATION_JOB_NOT_STARTABLE:{job.state}"
                )
            preparing = transition_pair_qualification_job(
                job,
                "preparing",
                progress_percent=max(10, job.progress_percent),
                cancel_requested=False,
            )
            snapshot = self.store.save_asset_pair_qualification_job(preparing)
            worker = threading.Thread(target=self._execute, args=(job_id,), daemon=True)
            self._workers[job_id] = worker
            try:
                worker.start()
            except Exception as error:
                self._workers.pop(job_id, None)
                paused = transition_pair_qualification_job(
                    preparing,
                    "paused",
                    progress_percent=preparing.progress_percent,
                    issue_codes=["ASSET_QUALIFICATION_WORKER_START_FAILED"],
                )
                self.store.save_asset_pair_qualification_job(paused)
                raise AssetQualificationServiceError(
                    "ASSET_QUALIFICATION_WORKER_START_FAILED"
                ) from error
            return snapshot

    # 功能：
    #   请求停止活动运行并保存暂停状态；恢复时使用新的尝试目录。
    # 输入：
    #   self：当前验收服务。
    #   job_id：需要暂停的任务标识。
    # 输出：
    #   record：已保存的暂停记录。
    def pause(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self.store.load_asset_pair_qualification_job(job_id)
            if job.state not in _INTERRUPTED_STATES:
                raise AssetQualificationServiceError(
                    f"ASSET_QUALIFICATION_JOB_NOT_PAUSABLE:{job.state}"
                )
            if job.state == "running":
                self.runtime_canceller(job_id)
            paused = transition_pair_qualification_job(
                job,
                "paused",
                progress_percent=job.progress_percent,
                cancel_requested=True,
            )
            record = self.store.save_asset_pair_qualification_job(paused)
            return record

    # 功能：
    #   请求停止活动运行，并将非终态任务永久取消，阻止晚到的结果发布。
    # 输入：
    #   self：当前验收服务。
    #   job_id：需要取消的任务标识。
    # 输出：
    #   record：已保存的取消记录。
    def cancel(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self.store.load_asset_pair_qualification_job(job_id)
            if job.state in _TERMINAL_STATES:
                raise AssetQualificationServiceError(
                    f"ASSET_QUALIFICATION_JOB_TERMINAL:{job.state}"
                )
            if job.state == "running":
                self.runtime_canceller(job_id)
            cancelled = transition_pair_qualification_job(
                job,
                "cancelled",
                progress_percent=job.progress_percent,
                cancel_requested=True,
            )
            record = self.store.save_asset_pair_qualification_job(cancelled)
            return record

    # 功能：
    #   返回已核对摘要的验收回执和资产契约摘要，不额外暴露宿主资产目录。
    # 输入：
    #   self：当前验收服务。
    #   job_id：已完成验收的任务标识。
    # 输出：
    #   evidence：验收身份、原始回执及运行契约摘要。
    def evidence(self, job_id: str) -> dict[str, object]:
        job = self.store.load_asset_pair_qualification_job(job_id)
        try:
            receipt, evidence_sha256 = self.store.verified_asset_pair_receipt(job)
        except (AssetImportError, KeyError, OSError, TypeError, ValueError) as error:
            raise AssetQualificationServiceError("ASSET_QUALIFICATION_EVIDENCE_INVALID") from error
        try:
            runtime_contracts = self._runtime_contracts(job)
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise AssetQualificationServiceError(
                "ASSET_QUALIFICATION_RUNTIME_CONTRACT_INVALID"
            ) from error
        evidence = {
            "schema_version": "dronedream.asset-qualification-evidence.v1",
            "job_id": job.job_id,
            "qualification_id": job.qualification_id,
            "map_asset_id": job.map_asset_id,
            "map_content_sha256": job.result_map_content_sha256,
            "vehicle_asset_id": job.vehicle_asset_id,
            "vehicle_content_sha256": job.result_vehicle_content_sha256,
            "evidence_sha256": evidence_sha256,
            "receipt": receipt.model_dump(mode="json"),
            "runtime_contracts": runtime_contracts,
        }
        return evidence

    # 功能：
    #   从已验收的确切内容版本提取地图范围和飞机能力摘要，不返回宿主路径。
    # 输入：
    #   self：当前验收服务。
    #   job：包含双方已验收内容摘要的任务。
    # 输出：
    #   contracts：地图与飞机的运行契约摘要。
    def _runtime_contracts(self, job: AssetPairQualificationJob) -> dict[str, object]:
        if job.result_map_content_sha256 is None or job.result_vehicle_content_sha256 is None:
            raise ValueError("qualified asset result hashes are missing")
        map_record = self.store.get_asset_version(job.map_asset_id, job.result_map_content_sha256)
        vehicle_record = self.store.get_asset_version(
            job.vehicle_asset_id, job.result_vehicle_content_sha256
        )
        map_selection = resolve_versioned_map(map_record)
        vehicle_selection = resolve_versioned_vehicle(vehicle_record)
        map_ir = AssetIR.model_validate(map_record["asset_ir"])
        vehicle_ir = AssetIR.model_validate(vehicle_record["asset_ir"])
        graph = MapAsset.model_validate(
            self._read_contract(map_selection.graph, map_selection.root, map_ir)
        )
        vehicle = VehicleAsset.model_validate(
            self._read_contract(
                vehicle_selection.vehicle_metadata, vehicle_selection.root, vehicle_ir
            )
        )
        if graph.asset_id != job.map_asset_id or vehicle.asset_id != job.vehicle_asset_id:
            raise ValueError("runtime contract identity changed")
        points = [node.position_m for node in graph.nodes]
        minimum = {axis: min(getattr(point, axis) for point in points) for axis in ("x", "y", "z")}
        maximum = {axis: max(getattr(point, axis) for point in points) for axis in ("x", "y", "z")}
        contracts = {
            "schema_version": "dronedream.asset-pair-runtime-contracts.v1",
            "map": {
                "asset_id": job.map_asset_id,
                "content_sha256": job.result_map_content_sha256,
                "coordinate_frame": "ENU",
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "named_entity_count": len(graph.named_entities),
                "navigation_bounds_m": {
                    "minimum": minimum,
                    "maximum": maximum,
                    "span": {axis: maximum[axis] - minimum[axis] for axis in ("x", "y", "z")},
                },
                "semantic_layers": list(map_ir.semantic_layers),
                "simulation_targets": [
                    target.model_dump(mode="json") for target in map_ir.simulation_targets
                ],
            },
            "vehicle": {
                **vehicle.model_dump(mode="json"),
                "content_sha256": job.result_vehicle_content_sha256,
                "vehicle_class": (
                    vehicle_ir.vehicle.vehicle_class if vehicle_ir.vehicle else "unknown"
                ),
                "simulation_targets": [
                    target.model_dump(mode="json") for target in vehicle_ir.simulation_targets
                ],
            },
        }
        return contracts

    # 功能：
    #   读取并核对同一份契约字节，避免先校验文件、再解析另一份被替换的内容。
    # 输入：
    #   path：已由运行时解析器选中的契约文件。
    #   root：对应资产版本根目录。
    #   asset_ir：该版本的规范化文件索引。
    # 输出：
    #   contract：通过大小、摘要和 JSON 边界检查的契约字典。
    @staticmethod
    def _read_contract(path: Path, root: Path, asset_ir: AssetIR) -> dict[str, object]:
        relative = path.relative_to(root).as_posix()
        entries = [entry for entry in asset_ir.files if entry.path == relative]
        if len(entries) != 1:
            raise AssetQualificationServiceError("ASSET_QUALIFICATION_CONTRACT_INDEX_INVALID")
        entry = entries[0]
        payload = read_plugin_file(path, limit=min(MAX_ASSET_IR_BYTES, entry.size_bytes))
        if len(payload) != entry.size_bytes or hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise AssetQualificationServiceError("ASSET_QUALIFICATION_CONTRACT_CHANGED")
        contract = decode_json(payload, limit=MAX_ASSET_IR_BYTES, node_limit=2_000_000)
        if not isinstance(contract, dict):
            raise AssetQualificationServiceError("ASSET_QUALIFICATION_CONTRACT_INVALID")
        return contract

    # 功能：
    #   获取独立、有效的运行环境快照，避免共享字典被原地修改后掩盖环境切换。
    # 输入：
    #   self：持有运行环境提供器的服务。
    # 输出：
    #   environment：当前实际运行环境的名称与版本映射。
    def _environment_snapshot(self) -> dict[str, str]:
        environment = self._evidence_snapshot(self.environment_versions_provider())
        if not 0 < len(environment) <= 64 or any(
            not key or not isinstance(value, str) or not value for key, value in environment.items()
        ):
            raise AssetQualificationServiceError("ASSET_QUALIFICATION_ENVIRONMENT_INVALID")
        return environment

    # 功能：
    #   对证据进行有界 JSON 检查并复制，隔离执行器、宿主和插件之间的可变容器。
    # 输入：
    #   value：跨回调边界传递的证据字典。
    # 输出：
    #   snapshot：不共享原始容器的 JSON 字典。
    @staticmethod
    def _evidence_snapshot(value: object) -> dict[str, object]:
        snapshot = decode_json(
            encode_json(value, limit=MAX_ASSET_IR_BYTES, node_limit=2_000_000),
            limit=MAX_ASSET_IR_BYTES,
            node_limit=2_000_000,
        )
        if not isinstance(snapshot, dict):
            raise AssetQualificationServiceError("ASSET_QUALIFICATION_EVIDENCE_INVALID")
        return snapshot

    # 功能：
    #   为运行桥接器提供实时取消判断，覆盖任务进入运行态但子进程尚未登记的窗口。
    # 输入：
    #   self：当前验收服务。
    #   job_id：正在交给桥接器启动的任务标识。
    # 输出：
    #   cancelled：任务是否已不再允许启动或继续运行。
    def _runtime_cancelled(self, job_id: str) -> bool:
        try:
            current = self.store.load_asset_pair_qualification_job(job_id)
            cancelled = current.state != "running" or current.cancel_requested
        except KeyError:
            cancelled = True
        return cancelled

    # 功能：
    #   1. 冻结双方资产，执行配对运行，验证核心门禁，再进行附加插件检查。
    #   2. 校验通过且任务仍活动时才发布双方结果；保留失败材料供排查。
    # 输入：
    #   self：当前验收服务。
    #   job_id：当前后台执行的任务标识。
    # 输出：
    #   None：不返回业务数据。
    def _execute(self, job_id: str) -> None:
        try:
            workspace, map_root, vehicle_root = self.store.asset_pair_qualification_paths(job_id)
            job = self.store.load_asset_pair_qualification_job(job_id)
            attempt = workspace / f"attempt-{job.revision:04d}"
            attempt.mkdir(parents=False)
            inputs = attempt / "inputs"
            inputs.mkdir()
            map_archive = self._archive_version(
                map_root,
                inputs / "map.ddpkg",
                expected_asset_id=job.map_asset_id,
                expected_content_sha256=job.map_content_sha256,
            )
            vehicle_archive = self._archive_version(
                vehicle_root,
                inputs / "vehicle.ddpkg",
                expected_asset_id=job.vehicle_asset_id,
                expected_content_sha256=job.vehicle_content_sha256,
            )
            prepared_root = attempt / "prepared"
            plan = prepare_asset_pair_qualification(
                map_archive=map_archive,
                vehicle_archive=vehicle_archive,
                work_root=prepared_root,
            )
            environment_versions = self._environment_snapshot()
            with self._lock:
                current = self.store.load_asset_pair_qualification_job(job_id)
                if current.state != "preparing":
                    return
                running = transition_pair_qualification_job(
                    current,
                    "running",
                    progress_percent=max(45, current.progress_percent),
                    qualification_id=plan.qualification_id,
                )
                self.store.save_asset_pair_qualification_job(running)
            runtime_root = attempt / "runtime-evidence"
            runtime_evidence = self.runtime_runner(
                job_id=job_id,
                work_root=prepared_root,
                run_dir=runtime_root,
                cancel_requested=lambda: self._runtime_cancelled(job_id),
            )
            runtime_evidence = self._evidence_snapshot(runtime_evidence)
            with self._lock:
                current = self.store.load_asset_pair_qualification_job(job_id)
                if current.state != "running":
                    return
                validating = transition_pair_qualification_job(
                    current,
                    "validating",
                    progress_percent=max(85, current.progress_percent),
                )
                self.store.save_asset_pair_qualification_job(validating)
            if self._environment_snapshot() != environment_versions:
                raise AssetQualificationServiceError("ASSET_QUALIFICATION_ENVIRONMENT_CHANGED")
            receipt = build_pair_qualification_receipt(
                plan=plan,
                work_root=prepared_root,
                runtime_evidence=runtime_evidence,
                environment_versions=environment_versions,
            )
            if self.qualification_checker is not None:
                map_asset_ir = inspect_ddpkg(map_archive).asset_ir
                vehicle_asset_ir = inspect_ddpkg(vehicle_archive).asset_ir
                plugin_evidence = self.qualification_checker(
                    plan=plan.model_dump(mode="json"),
                    map_asset_ir=map_asset_ir.model_dump(mode="json"),
                    vehicle_asset_ir=vehicle_asset_ir.model_dump(mode="json"),
                    runtime_evidence=self._evidence_snapshot(runtime_evidence),
                    environment_versions=dict(environment_versions),
                )
                plugin_evidence = self._evidence_snapshot(plugin_evidence)
                try:
                    plugin_snapshot = plugin_evidence["plugin_snapshot"]
                    plugin_snapshot_sha256 = plugin_evidence["plugin_snapshot_sha256"]
                    plugin_checks = plugin_evidence["plugin_checks"]
                    plugin_hook_receipts = plugin_evidence["plugin_hook_receipts"]
                except (KeyError, TypeError) as error:
                    raise AssetQualificationServiceError(
                        "ASSET_QUALIFICATION_PLUGIN_EVIDENCE_INVALID"
                    ) from error
                if (
                    not isinstance(plugin_snapshot, dict)
                    or not isinstance(plugin_snapshot_sha256, str)
                    or not isinstance(plugin_checks, list)
                    or not isinstance(plugin_hook_receipts, list)
                ):
                    raise AssetQualificationServiceError(
                        "ASSET_QUALIFICATION_PLUGIN_EVIDENCE_INVALID"
                    )
                receipt = build_pair_qualification_receipt(
                    plan=plan,
                    work_root=prepared_root,
                    runtime_evidence=runtime_evidence,
                    environment_versions=environment_versions,
                    plugin_snapshot=plugin_snapshot,
                    plugin_snapshot_sha256=plugin_snapshot_sha256,
                    plugin_checks=plugin_checks,
                    plugin_hook_receipts=plugin_hook_receipts,
                    qualified_at=receipt.qualified_at,
                )
            if self._environment_snapshot() != environment_versions:
                raise AssetQualificationServiceError("ASSET_QUALIFICATION_ENVIRONMENT_CHANGED")
            outputs = attempt / "outputs"
            outputs.mkdir()
            qualified_map = bind_qualification_receipt(
                source_archive=map_archive,
                destination_archive=outputs / "map-qualified.ddpkg",
                receipt=receipt,
            )
            qualified_vehicle = bind_qualification_receipt(
                source_archive=vehicle_archive,
                destination_archive=outputs / "vehicle-qualified.ddpkg",
                receipt=receipt,
            )
            with self._lock:
                current = self.store.load_asset_pair_qualification_job(job_id)
                if current.state != "validating":
                    return
                map_row, vehicle_row = self.asset_import_service.promote_locally_verified_pair(
                    map_source=qualified_map,
                    map_asset_id=job.map_asset_id,
                    vehicle_source=qualified_vehicle,
                    vehicle_asset_id=job.vehicle_asset_id,
                    environment_versions=environment_versions,
                    operation_id=job_id,
                )
                qualified = transition_pair_qualification_job(
                    current,
                    "qualified",
                    progress_percent=100,
                    result_map_content_sha256=str(map_row["content_sha256"]),
                    result_vehicle_content_sha256=str(vehicle_row["content_sha256"]),
                )
                self.store.save_asset_pair_qualification_job(qualified)
                self.asset_import_service.finalize_qualified_admissions(
                    asset_id=job.map_asset_id,
                    source_content_sha256=job.map_content_sha256,
                    result_content_sha256=str(map_row["content_sha256"]),
                )
                self.asset_import_service.finalize_qualified_admissions(
                    asset_id=job.vehicle_asset_id,
                    source_content_sha256=job.vehicle_content_sha256,
                    result_content_sha256=str(vehicle_row["content_sha256"]),
                )
        except Exception as error:
            self._fail_if_active(job_id, error)
        finally:
            with self._lock:
                self._workers.pop(job_id, None)

    # 功能：
    #   只将仍活动的任务保存为失败，不覆盖用户暂停、取消或已经发布的结果。
    # 输入：
    #   self：当前验收服务。
    #   job_id：发生异常的任务标识。
    #   error：本次执行的原始异常。
    # 输出：
    #   None：不返回业务数据。
    def _fail_if_active(self, job_id: str, error: Exception) -> None:
        with self._lock:
            try:
                current = self.store.load_asset_pair_qualification_job(job_id)
            except KeyError:
                return
            if current.state in _TERMINAL_STATES or current.state == "paused":
                return
            failed = transition_pair_qualification_job(
                current,
                "failed",
                progress_percent=current.progress_percent,
                issue_codes=[self._public_issue_code(error)],
            )
            self.store.save_asset_pair_qualification_job(failed)

    # 功能：
    #   只公开长度和格式受限的领域错误码，隐藏异常正文中的路径或其他细节。
    # 输入：
    #   error：底层执行或存储异常。
    # 输出：
    #   code：可保存到任务记录中的稳定错误码。
    @staticmethod
    def _public_issue_code(error: Exception) -> str:
        code = "ASSET_QUALIFICATION_INTERNAL"
        if isinstance(
            error,
            (
                AssetPairQualificationError,
                AssetQualificationServiceError,
                AssetImportError,
                AssetPackageError,
            ),
        ) or error.__class__.__name__ in {"RuntimeBridgeError", "PluginManagerError"}:
            candidate = str(error).split(":", 1)[0]
            if re.fullmatch(
                r"(?:ASSET|DDPKG|RUNTIME|DRONEDREAM|PLUGIN)_[A-Z0-9_]{1,80}", candidate
            ):
                code = candidate
        if isinstance(error, OSError):
            code = "ASSET_QUALIFICATION_IO_FAILURE"
        return code

    # 功能：
    #   使用共享的有界、不可覆盖导出链路重新冻结已安装版本，供本次验收使用。
    # 输入：
    #   source_root：已安装资产内容目录。
    #   destination：本次尝试独占的新归档路径。
    #   expected_asset_id：任务绑定的资产标识。
    #   expected_content_sha256：任务绑定的内容摘要。
    # 输出：
    #   archive：完整验证后发布的归档路径。
    @staticmethod
    def _archive_version(
        source_root: Path,
        destination: Path,
        *,
        expected_asset_id: str,
        expected_content_sha256: str,
    ) -> Path:
        try:
            archive = export_stored_asset(
                source_root,
                destination,
                expected_asset_id=expected_asset_id,
                expected_content_sha256=expected_content_sha256,
            )
        except (OSError, ValueError) as error:
            raise AssetQualificationServiceError(
                "ASSET_QUALIFICATION_STORED_EXPORT_INVALID"
            ) from error
        return archive
