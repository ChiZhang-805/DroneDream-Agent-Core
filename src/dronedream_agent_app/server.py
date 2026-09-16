"""Authenticated loopback API hosted inside the packaged desktop sidecar."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import sqlite3
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import anyio
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.background import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from dronedream_agent_core.asset_packages import AssetPackageError
from dronedream_agent_core.asset_policy import asset_qualification_policy
from dronedream_agent_core.model_harness.design import (
    HarnessEditOperation,
    HarnessTopologyCandidate,
)
from dronedream_agent_core.model_harness.memory import (
    MemoryOwnerScope,
    default_account_memory_path,
)
from dronedream_agent_core.model_harness.memory_projection import (
    MemoryProjectionCredentials,
    MemoryProjectionSyncStatus,
)
from dronedream_agent_core.model_harness.model_port import ModelInvocationError
from dronedream_agent_core.orchestrator import (
    MissionClarificationRequired,
    MissionPreparationBlocked,
)
from dronedream_agent_core.plugin_contracts import (
    PluginGovernancePolicy,
    PluginMarketplaceSource,
)

from . import __version__
from .asset_import_service import AssetImportService
from .asset_issue_catalog import asset_job_issue_report
from .asset_qualification_service import (
    AssetQualificationService,
    AssetQualificationServiceError,
)
from .asset_remote_sources import RemoteAssetSourceService
from .connector_credentials import ConnectorCredentialService
from .custom_models import CredentialVault, CustomModelService, ModelConnection
from .harness_design_service import HarnessDesignService, HarnessDesignServiceError
from .identity import SupabaseJwtVerifier, VerifiedIdentity
from .mission_service import MissionService, _validate_gateway
from .models import (
    AccountMemoryCandidateDecisionRequest,
    AccountMemoryCandidateListRequest,
    AccountMemoryForgetRequest,
    AccountMemoryScopeRequest,
    AssetInterpretationRequest,
    AssetPairQualificationCreateRequest,
    AssetRemoteImportRequest,
    ConnectorCredentialCreateRequest,
    CustomModelCreateRequest,
    CustomModelDiscoverRequest,
    HarnessRevisionActionRequest,
    MessageCreate,
    MissionExecuteRequest,
    MissionPrepareRequest,
    OperatorControlRequest,
    OperatorTakeoverGrantRequest,
    PluginConfigurationRequest,
    PluginMarketplaceInstallRequest,
    PluginRollbackRequest,
    RuntimeMessageRequest,
    SettingsPatch,
    ThreadCreate,
    ThreadPatch,
    TrustedPublisherRequest,
)
from .plugin_manager import PluginManager, PluginManagerError
from .plugin_marketplace import PluginMarketplaceError, PluginMarketplaceService
from .runtime_manager import RuntimeBridgeError, RuntimeManager
from .storage import AppStore, AssetImportError


async def _stage_upload(
    upload: UploadFile,
    destination: Path,
    *,
    maximum_bytes: int,
    too_large_detail: str,
) -> int:
    """Stream an upload without blocking the API event loop on filesystem writes."""

    size = 0
    async with await anyio.open_file(destination, "wb") as output:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > maximum_bytes:
                raise HTTPException(status_code=413, detail=too_large_detail)
            await output.write(chunk)
    return size


def _projection_credentials(
    identity: VerifiedIdentity,
    identity_token: str | None,
    publishable_key: str | None,
) -> tuple[MemoryProjectionCredentials | None, str | None]:
    if not identity_token or not publishable_key:
        return None, "MEMORY_PROJECTION_CONFIGURATION_MISSING"
    suffix = "/auth/v1"
    if not identity.issuer.endswith(suffix):
        return None, "MEMORY_PROJECTION_ISSUER_INVALID"
    try:
        return (
            MemoryProjectionCredentials(
                project_url=identity.issuer[: -len(suffix)],
                publishable_key=publishable_key,
                access_token=identity_token,
                timeout_seconds=2.0,
            ),
            None,
        )
    except ValueError as error:
        return None, f"MEMORY_PROJECTION_CONFIGURATION_INVALID:{error}"[:240]


def create_app(
    *,
    store: AppStore,
    token: str,
    resource_root: Path | None = None,
    connector_credential_vault: CredentialVault | None = None,
    plugin_isolator_path: Path | None = None,
    account_memory_path: Path | None = None,
    identity_verifier: Callable[[str], VerifiedIdentity] | None = None,
) -> FastAPI:
    official_plugins_root = (
        resource_root / "official-plugins" if resource_root is not None else None
    )
    plugin_manager = PluginManager(
        store,
        official_plugins_root=official_plugins_root,
        plugin_isolator_path=plugin_isolator_path,
    )
    plugin_marketplace = PluginMarketplaceService(store, plugin_manager)
    harness_design = HarnessDesignService(store, plugin_manager)
    custom_models = CustomModelService(store)
    connector_credentials = ConnectorCredentialService(store, connector_credential_vault)
    mission_service = MissionService(
        store,
        plugin_manager,
        connector_credentials,
        harness_design,
        account_memory_path,
    )
    verify_identity = identity_verifier or SupabaseJwtVerifier()
    runtime_manager = RuntimeManager(store, resource_root, plugin_manager)
    asset_import_service = AssetImportService(
        store,
        environment_versions_provider=runtime_manager.qualification_environment_versions,
        source_detector=plugin_manager.detect_asset_source_with_plugins,
        plugin_source_normalizer=plugin_manager.normalize_asset_source_with_plugin,
    )
    if resource_root is not None:
        bundled_assets = resource_root / "default-assets"
        asset_import_service.seed_bundled_sources(bundled_assets)
    remote_asset_sources = RemoteAssetSourceService(store.root / "remote-asset-staging")
    asset_qualification_service = AssetQualificationService(
        store,
        asset_import_service,
        runtime_runner=runtime_manager.run_asset_pair_qualification,
        runtime_canceller=runtime_manager.cancel_asset_pair_qualification,
        environment_versions_provider=runtime_manager.qualification_environment_versions,
        qualification_checker=plugin_manager.run_asset_qualification_checks,
    )
    plugin_manager.set_disable_guard(runtime_manager.prepare_plugin_disable)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            plugin_manager.close()

    app = FastAPI(
        title="DroneDream AGENT Core",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["tauri://localhost", "http://tauri.localhost", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-DroneDream-Identity-Token",
            "X-DroneDream-Supabase-Publishable-Key",
        ],
    )

    def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {token}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="LOCAL_SESSION_REQUIRED"
            )

    local = Depends(authorize)

    def authenticate_identity(
        identity_token: str | None = Header(default=None, alias="X-DroneDream-Identity-Token"),
    ) -> VerifiedIdentity:
        if not identity_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="VERIFIED_IDENTITY_REQUIRED",
            )
        try:
            return verify_identity(identity_token)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(error),
            ) from error

    verified_identity = Depends(authenticate_identity)

    def require_identity_match(
        payload: AccountMemoryScopeRequest
        | MissionPrepareRequest
        | MissionExecuteRequest
        | AssetInterpretationRequest,
        identity: VerifiedIdentity,
    ) -> None:
        mismatched = (
            (
                payload.expected_owner_account_id is not None
                and payload.expected_owner_account_id != identity.owner_account_id
            )
            or (
                payload.expected_tenant_id is not None
                and payload.expected_tenant_id != identity.tenant_id
            )
            or (
                payload.expected_organization_id is not None
                and payload.expected_organization_id != identity.organization_id
            )
        )
        if mismatched:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="VERIFIED_IDENTITY_SCOPE_MISMATCH",
            )

    def require_account_memory_scope(
        payload: AccountMemoryScopeRequest,
        identity: VerifiedIdentity,
    ) -> MemoryOwnerScope:
        require_identity_match(payload, identity)
        try:
            store.get_thread(payload.thread_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="THREAD_NOT_FOUND") from error
        scope = MemoryOwnerScope(
            owner_account_id=identity.owner_account_id,
            tenant_id=identity.tenant_id,
            organization_id=identity.organization_id,
            source_edition=payload.source_edition,
        )
        try:
            mission_service.account_memory.require_thread_binding(scope, payload.thread_id)
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="ACCOUNT_MEMORY_THREAD_NOT_BOUND",
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="ACCOUNT_MEMORY_THREAD_OWNER_MISMATCH",
            ) from error
        return scope

    def sync_account_memory_projection(
        scope: MemoryOwnerScope,
        identity: VerifiedIdentity,
        identity_token: str | None,
        publishable_key: str | None,
        *,
        local_memory_enabled: bool | None = None,
        additional_issues: tuple[str, ...] = (),
    ) -> dict[str, object]:
        credentials, configuration_issue = _projection_credentials(
            identity, identity_token, publishable_key
        )
        enabled = (
            bool(store.get_settings().get("memory_enabled", True))
            if local_memory_enabled is None
            else local_memory_enabled
        )
        try:
            projection = mission_service.memory_projection.sync(
                scope,
                credentials,
                local_memory_enabled=enabled,
                operation_limit=8,
                pull_limit=32,
                time_budget_seconds=3.0,
            )
        except (OSError, ValueError, sqlite3.Error) as error:
            projection = MemoryProjectionSyncStatus(
                status="conflict",
                pending_operations=mission_service.memory_projection.pending_count(scope),
                issues=(f"MEMORY_PROJECTION_SYNC_REJECTED:{error}"[:240],),
            )
        issues = list(projection.issues)
        if configuration_issue:
            issues.append(configuration_issue)
        issues.extend(additional_issues)
        if issues != list(projection.issues):
            projection = projection.model_copy(update={"issues": tuple(issues[:32])})
        return projection.model_dump(mode="json")

    def require_runtime_thread_scope(
        thread_id: str,
        identity: VerifiedIdentity,
        *,
        source_edition: str = "autonomy",
    ) -> MemoryOwnerScope:
        """Authorize every runtime read/write against the same bound owner scope."""

        try:
            store.get_thread(thread_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="THREAD_NOT_FOUND") from error
        scope = MemoryOwnerScope(
            owner_account_id=identity.owner_account_id,
            tenant_id=identity.tenant_id,
            organization_id=identity.organization_id,
            source_edition=source_edition,
        )
        try:
            mission_service.account_memory.require_thread_binding(scope, thread_id)
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="EXECUTION_THREAD_NOT_BOUND",
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="EXECUTION_THREAD_OWNER_MISMATCH",
            ) from error
        return scope

    def require_custom_model_connector() -> None:
        if not plugin_manager.model_provider_enabled("custom"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="CUSTOM_MODEL_PROVIDER_PLUGIN_DISABLED",
            )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ready", "version": __version__}

    @app.post("/shutdown", dependencies=[local], status_code=202)
    def shutdown() -> dict[str, object]:
        runtime_manager.shutdown()

        def exit_after_response() -> None:
            time.sleep(0.15)
            os._exit(0)

        threading.Thread(target=exit_after_response, daemon=True).start()
        return {"accepted": True}

    # 功能：
    #   为桌面和命令行提供同一活动模型目录，不必重复传输所有插件配置和历史任务。
    # 输入：
    #   无。
    # 输出：
    #   catalog：当前可用的默认及自定义模型。
    @app.get("/v1/models", dependencies=[local])
    def model_catalog() -> dict[str, object]:
        default_models = plugin_manager.model_catalog()
        custom_catalog = (
            custom_models.catalog() if plugin_manager.model_provider_enabled("custom") else []
        )
        catalog = {"models": [*default_models, *custom_catalog]}
        return catalog

    @app.get("/v1/bootstrap", dependencies=[local])
    def bootstrap() -> dict[str, object]:
        return {
            "models": model_catalog()["models"],
            "threads": store.list_threads(),
            "asset_import_jobs": store.list_asset_import_jobs(),
            "asset_versions": store.list_asset_versions(),
            "asset_qualification_jobs": store.list_asset_pair_qualification_jobs(),
            "asset_source_adapters": plugin_manager.source_adapter_catalog(),
            "plugins": plugin_manager.list_plugins(),
            "connector_credentials": connector_credentials.list(),
            "settings": store.get_settings(),
        }

    @app.get("/v1/connector-credentials", dependencies=[local])
    def list_connector_credentials() -> list[dict[str, object]]:
        return connector_credentials.list()

    @app.post("/v1/connector-credentials", dependencies=[local], status_code=201)
    def create_connector_credential(
        payload: ConnectorCredentialCreateRequest,
    ) -> dict[str, object]:
        try:
            return connector_credentials.create(**payload.model_dump())
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.delete("/v1/connector-credentials/{reference}", dependencies=[local])
    def delete_connector_credential(reference: str) -> dict[str, object]:
        try:
            connector_credentials.delete(reference)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="CONNECTOR_CREDENTIAL_NOT_FOUND") from error
        return {"deleted": True, "reference": reference}

    @app.post("/v1/threads", dependencies=[local], status_code=201)
    def create_thread(payload: ThreadCreate) -> dict[str, object]:
        title = payload.title or ("New task" if payload.locale == "en-US" else "新任务")
        return store.create_thread(title, payload.selected_model, payload.locale)

    @app.get("/v1/threads/{thread_id}", dependencies=[local])
    def get_thread(thread_id: str) -> dict[str, object]:
        try:
            return store.get_thread(thread_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="THREAD_NOT_FOUND") from error

    @app.post("/v1/account-memory/candidates/list", dependencies=[local])
    def list_account_memory_candidates(
        payload: AccountMemoryCandidateListRequest,
        identity: VerifiedIdentity = verified_identity,
    ) -> dict[str, object]:
        scope = require_account_memory_scope(payload, identity)
        try:
            candidates = mission_service.account_memory.list_candidates(
                scope,
                memory_key=payload.memory_key,
                limit=payload.limit,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "namespace": scope.namespace,
            "owner_bound": True,
            "thread_bound": True,
            "candidates": candidates,
        }

    @app.post(
        "/v1/account-memory/candidates/{candidate_id}/resolve",
        dependencies=[local],
    )
    def resolve_account_memory_candidate(
        candidate_id: str,
        payload: AccountMemoryCandidateDecisionRequest,
        identity_token: str | None = Header(default=None, alias="X-DroneDream-Identity-Token"),
        publishable_key: str | None = Header(
            default=None, alias="X-DroneDream-Supabase-Publishable-Key"
        ),
        identity: VerifiedIdentity = verified_identity,
    ) -> dict[str, object]:
        scope = require_account_memory_scope(payload, identity)
        projection_issues: list[str] = []
        try:
            # Legacy candidates created before the projection bridge are staged
            # before their explicit local decision is mirrored.
            mission_service.memory_projection.queue_candidate(scope, candidate_id)
        except (KeyError, OSError, ValueError, sqlite3.Error) as error:
            projection_issues.append(f"MEMORY_PROJECTION_CANDIDATE_QUEUE_REJECTED:{error}"[:240])
        try:
            memory = mission_service.account_memory.resolve_candidate(
                scope,
                candidate_id,
                approve=True,
                explicit_reconsent=payload.explicit_reconsent,
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="ACCOUNT_MEMORY_CANDIDATE_NOT_FOUND"
            ) from error
        except ValueError as error:
            detail = str(error)
            code = (
                status.HTTP_409_CONFLICT
                if detail == "ACCOUNT_MEMORY_CANDIDATE_NOT_PENDING"
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            raise HTTPException(status_code=code, detail=detail) from error
        if memory is None:  # pragma: no cover - approve=True is fixed above
            raise HTTPException(status_code=409, detail="ACCOUNT_MEMORY_RESOLUTION_FAILED")
        try:
            mission_service.memory_projection.queue_resolution(scope, candidate_id, approve=True)
        except (OSError, ValueError, sqlite3.Error) as error:
            projection_issues.append(f"MEMORY_PROJECTION_RESOLUTION_QUEUE_REJECTED:{error}"[:240])
        projection = sync_account_memory_projection(
            scope,
            identity,
            identity_token,
            publishable_key,
            additional_issues=tuple(projection_issues),
        )
        return {
            "candidate_id": candidate_id,
            "status": "consolidated",
            "memory": memory.model_dump(mode="json"),
            "memory_projection": projection,
        }

    @app.post(
        "/v1/account-memory/candidates/{candidate_id}/reject",
        dependencies=[local],
    )
    def reject_account_memory_candidate(
        candidate_id: str,
        payload: AccountMemoryCandidateDecisionRequest,
        identity_token: str | None = Header(default=None, alias="X-DroneDream-Identity-Token"),
        publishable_key: str | None = Header(
            default=None, alias="X-DroneDream-Supabase-Publishable-Key"
        ),
        identity: VerifiedIdentity = verified_identity,
    ) -> dict[str, object]:
        scope = require_account_memory_scope(payload, identity)
        projection_issues: list[str] = []
        try:
            mission_service.memory_projection.queue_candidate(scope, candidate_id)
        except (KeyError, OSError, ValueError, sqlite3.Error) as error:
            projection_issues.append(f"MEMORY_PROJECTION_CANDIDATE_QUEUE_REJECTED:{error}"[:240])
        try:
            mission_service.account_memory.reject_candidate(scope, candidate_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="ACCOUNT_MEMORY_CANDIDATE_NOT_FOUND"
            ) from error
        except ValueError as error:
            detail = str(error)
            code = (
                status.HTTP_409_CONFLICT
                if detail == "ACCOUNT_MEMORY_CANDIDATE_NOT_PENDING"
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            raise HTTPException(status_code=code, detail=detail) from error
        try:
            mission_service.memory_projection.queue_resolution(scope, candidate_id, approve=False)
        except (OSError, ValueError, sqlite3.Error) as error:
            projection_issues.append(f"MEMORY_PROJECTION_RESOLUTION_QUEUE_REJECTED:{error}"[:240])
        projection = sync_account_memory_projection(
            scope,
            identity,
            identity_token,
            publishable_key,
            additional_issues=tuple(projection_issues),
        )
        return {
            "candidate_id": candidate_id,
            "status": "rejected",
            "memory_projection": projection,
        }

    @app.post("/v1/account-memory/forget", dependencies=[local])
    def forget_account_memory(
        payload: AccountMemoryForgetRequest,
        identity_token: str | None = Header(default=None, alias="X-DroneDream-Identity-Token"),
        publishable_key: str | None = Header(
            default=None, alias="X-DroneDream-Supabase-Publishable-Key"
        ),
        identity: VerifiedIdentity = verified_identity,
    ) -> dict[str, object]:
        scope = require_account_memory_scope(payload, identity)
        try:
            result = mission_service.account_memory.forget(
                scope,
                payload.memory_key,
                mode=payload.mode,
                reason=payload.reason,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        projection_issues: list[str] = []
        try:
            mission_service.memory_projection.queue_forget(
                scope, payload.memory_key, mode=payload.mode
            )
        except (OSError, ValueError, sqlite3.Error) as error:
            projection_issues.append(f"MEMORY_PROJECTION_FORGET_QUEUE_REJECTED:{error}"[:240])
        projection = sync_account_memory_projection(
            scope,
            identity,
            identity_token,
            publishable_key,
            additional_issues=tuple(projection_issues),
        )
        return {"status": "forgotten", **result, "memory_projection": projection}

    @app.patch("/v1/threads/{thread_id}", dependencies=[local])
    def patch_thread(thread_id: str, payload: ThreadPatch) -> dict[str, object]:
        try:
            return store.patch_thread(thread_id, payload.model_dump(exclude_unset=True))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="THREAD_NOT_FOUND") from error

    @app.post("/v1/threads/{thread_id}/messages", dependencies=[local], status_code=201)
    def append_message(thread_id: str, payload: MessageCreate) -> dict[str, object]:
        try:
            return store.append_message(
                thread_id,
                role=payload.role,
                kind=payload.kind,
                content=payload.content,
                metadata=payload.metadata,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="THREAD_NOT_FOUND") from error

    @app.post("/v1/threads/{thread_id}/attachments", dependencies=[local], status_code=201)
    async def upload_attachment(
        thread_id: str,
        attachment: Annotated[UploadFile, File()],
    ) -> dict[str, object]:
        descriptor, temporary_name = tempfile.mkstemp(prefix="dd-attachment-")
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            await _stage_upload(
                attachment,
                temporary,
                maximum_bytes=25 * 1024 * 1024,
                too_large_detail="ATTACHMENT_TOO_LARGE",
            )
            return store.save_attachment(
                thread_id,
                display_name=attachment.filename or "attachment",
                content_type=attachment.content_type or "application/octet-stream",
                source=temporary,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="THREAD_NOT_FOUND") from error
        finally:
            await anyio.Path(temporary).unlink(missing_ok=True)

    @app.get("/v1/asset-import-jobs", dependencies=[local])
    def list_asset_import_jobs() -> list[dict[str, object]]:
        return store.list_asset_import_jobs()

    @app.get("/v1/asset-source-adapters", dependencies=[local])
    def list_asset_source_adapters() -> list[dict[str, object]]:
        return plugin_manager.source_adapter_catalog()

    @app.get("/v1/asset-qualification-policy", dependencies=[local])
    def get_asset_qualification_policy() -> dict[str, object]:
        return asset_qualification_policy()

    # 功能：
    #   导出指定内容版本，使用本次独占目录保存下载文件，响应结束后只清理自身文件。
    # 输入：
    #   asset_id：用户请求导出的资产标识。
    #   content_sha256：需要导出的完整内容摘要。
    #   background_tasks：响应完成后的清理任务队列。
    # 输出：
    #   response：指向已验证资产包的下载响应。
    @app.get(
        "/v1/asset-versions/{asset_id}/{content_sha256}/package",
        dependencies=[local],
        response_class=FileResponse,
    )
    def export_asset_version(
        asset_id: str,
        content_sha256: str,
        background_tasks: BackgroundTasks,
    ) -> FileResponse:
        if len(content_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in content_sha256
        ):
            raise HTTPException(status_code=400, detail="ASSET_CONTENT_SHA256_INVALID")
        directory = Path(tempfile.mkdtemp(prefix="ddpkg-export-", dir=store.root))
        directory_identity = directory.stat()
        temporary = directory / "package.ddpkg"
        file_identity = None

        # 功能：
        #   只删除本次完成导出的同一文件和空目录，保留碰撞写入或身份已变化的内容。
        # 输入：
        #   无：使用当前请求持有的目录与文件身份。
        # 输出：
        #   None：不返回业务数据。
        def cleanup_export() -> None:
            try:
                if directory.resolve().parent != store.root.resolve() or not os.path.samestat(
                    directory_identity, directory.stat(follow_symlinks=False)
                ):
                    return
                if file_identity is not None and os.path.samestat(
                    file_identity, temporary.stat(follow_symlinks=False)
                ):
                    temporary.unlink()
                # 失败时不拥有最终文件；非空目录留下供排查，不递归删除未知内容。
                directory.rmdir()
            except OSError:
                return

        try:
            asset_import_service.export_version(
                asset_id=asset_id,
                content_sha256=content_sha256,
                destination=temporary,
            )
            file_identity = temporary.stat(follow_symlinks=False)
        except KeyError as error:
            cleanup_export()
            raise HTTPException(status_code=404, detail="ASSET_VERSION_NOT_FOUND") from error
        except AssetImportError as error:
            cleanup_export()
            raise HTTPException(status_code=422, detail=str(error)) from error
        except BaseException:
            cleanup_export()
            raise
        background_tasks.add_task(cleanup_export)
        response = FileResponse(
            temporary,
            media_type="application/vnd.dronedream.ddpkg+zip",
            filename=f"{asset_id}-{content_sha256[:12]}.ddpkg",
        )
        return response

    @app.get("/v1/asset-import-jobs/{job_id}", dependencies=[local])
    def get_asset_import_job(job_id: str) -> dict[str, object]:
        try:
            return store.get_asset_import_job(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="ASSET_IMPORT_JOB_NOT_FOUND") from error

    @app.get("/v1/asset-import-jobs/{job_id}/issues", dependencies=[local])
    def get_asset_import_job_issues(job_id: str) -> dict[str, object]:
        try:
            return asset_job_issue_report(store.get_asset_import_job(job_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="ASSET_IMPORT_JOB_NOT_FOUND") from error

    @app.post("/v1/asset-import-jobs", dependencies=[local], status_code=202)
    async def create_asset_import_job(
        bundle: Annotated[UploadFile, File()],
        source_format: Annotated[str, Form()] = "auto",
        expected_kind: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        if (
            not source_format
            or len(source_format) > 80
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
                for character in source_format
            )
        ):
            raise HTTPException(status_code=400, detail="ASSET_SOURCE_FORMAT_INVALID")
        if expected_kind not in {None, "map", "world", "vehicle"}:
            raise HTTPException(status_code=400, detail="ASSET_KIND_INVALID")
        suffix = Path(bundle.filename or "source.upload").suffix.casefold()
        if (
            not suffix
            or len(suffix) > 16
            or any(character not in ".abcdefghijklmnopqrstuvwxyz0123456789" for character in suffix)
        ):
            suffix = ".upload"
        descriptor, temporary_name = tempfile.mkstemp(prefix="dd-asset-source-", suffix=suffix)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            await _stage_upload(
                bundle,
                temporary,
                maximum_bytes=8 * 1024 * 1024 * 1024,
                too_large_detail="UPLOAD_TOO_LARGE",
            )
            return asset_import_service.create(
                source=temporary,
                source_name=bundle.filename or "asset.ddpkg",
                source_format=source_format,
                expected_kind=expected_kind,
            )
        except AssetImportError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            await anyio.Path(temporary).unlink(missing_ok=True)

    @app.post("/v1/asset-import-jobs/remote", dependencies=[local], status_code=202)
    def create_remote_asset_import_job(
        request: AssetRemoteImportRequest,
    ) -> dict[str, object]:
        try:
            with remote_asset_sources.acquire(
                source_type=request.source_type,
                location=str(request.location),
                expected_sha256=request.expected_sha256,
                git_ref=request.git_ref,
                subpath=request.subpath,
            ) as (source, source_name):
                return asset_import_service.create(
                    source=source,
                    source_name=source_name,
                    source_format=request.source_format,
                    expected_kind=request.expected_kind,
                )
        except AssetImportError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/asset-import-jobs/{job_id}/process", dependencies=[local])
    def process_asset_import_job(job_id: str) -> dict[str, object]:
        try:
            return asset_import_service.process(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="ASSET_IMPORT_JOB_NOT_FOUND") from error
        except AssetImportError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post(
        "/v1/asset-import-jobs/{job_id}/companion-result",
        dependencies=[local],
    )
    async def submit_asset_import_companion_result(
        job_id: str,
        source_package_sha256: Annotated[str, Form()],
        adapter_id: Annotated[str, Form()],
        result: Annotated[UploadFile, File()],
    ) -> dict[str, object]:
        descriptor, temporary_name = tempfile.mkstemp(prefix="dd-companion-", suffix=".ddpkg")
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            await _stage_upload(
                result,
                temporary,
                maximum_bytes=8 * 1024 * 1024 * 1024,
                too_large_detail="UPLOAD_TOO_LARGE",
            )
            return asset_import_service.submit_companion_result(
                job_id,
                result=temporary,
                source_package_sha256=source_package_sha256,
                adapter_id=adapter_id,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="ASSET_IMPORT_JOB_NOT_FOUND") from error
        except (AssetImportError, AssetPackageError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            await anyio.Path(temporary).unlink(missing_ok=True)

    @app.post("/v1/asset-import-jobs/{job_id}/cancel", dependencies=[local])
    def cancel_asset_import_job(job_id: str) -> dict[str, object]:
        try:
            return asset_import_service.cancel(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="ASSET_IMPORT_JOB_NOT_FOUND") from error
        except AssetImportError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/v1/asset-qualification-jobs", dependencies=[local])
    def list_asset_qualification_jobs() -> list[dict[str, object]]:
        return store.list_asset_pair_qualification_jobs()

    @app.get("/v1/asset-qualification-jobs/{job_id}", dependencies=[local])
    def get_asset_qualification_job(job_id: str) -> dict[str, object]:
        try:
            return store.get_asset_pair_qualification_job(job_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="ASSET_QUALIFICATION_JOB_NOT_FOUND"
            ) from error

    @app.get("/v1/asset-qualification-jobs/{job_id}/issues", dependencies=[local])
    def get_asset_qualification_job_issues(job_id: str) -> dict[str, object]:
        try:
            return asset_job_issue_report(store.get_asset_pair_qualification_job(job_id))
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="ASSET_QUALIFICATION_JOB_NOT_FOUND"
            ) from error

    @app.get("/v1/asset-qualification-jobs/{job_id}/evidence", dependencies=[local])
    def get_asset_qualification_evidence(job_id: str) -> dict[str, object]:
        try:
            return asset_qualification_service.evidence(job_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="ASSET_QUALIFICATION_JOB_NOT_FOUND"
            ) from error
        except AssetQualificationServiceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/asset-qualification-jobs", dependencies=[local], status_code=201)
    def create_asset_qualification_job(
        payload: AssetPairQualificationCreateRequest,
    ) -> dict[str, object]:
        try:
            return asset_qualification_service.create(**payload.model_dump())
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="ASSET_QUALIFICATION_ASSET_VERSION_NOT_FOUND"
            ) from error
        except AssetQualificationServiceError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/asset-qualification-jobs/{job_id}/start", dependencies=[local], status_code=202)
    def start_asset_qualification_job(job_id: str) -> dict[str, object]:
        try:
            return asset_qualification_service.start(job_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="ASSET_QUALIFICATION_JOB_NOT_FOUND"
            ) from error
        except AssetQualificationServiceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/asset-qualification-jobs/{job_id}/pause", dependencies=[local])
    def pause_asset_qualification_job(job_id: str) -> dict[str, object]:
        try:
            return asset_qualification_service.pause(job_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="ASSET_QUALIFICATION_JOB_NOT_FOUND"
            ) from error
        except AssetQualificationServiceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/asset-qualification-jobs/{job_id}/cancel", dependencies=[local])
    def cancel_asset_qualification_job(job_id: str) -> dict[str, object]:
        try:
            return asset_qualification_service.cancel(job_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="ASSET_QUALIFICATION_JOB_NOT_FOUND"
            ) from error
        except AssetQualificationServiceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    # 功能：
    #   在真实账户及模型授权下解析指定资产并保存可复用理解，不创建计划或启动飞行。
    # 输入：
    #   thread_id：承载调用记录的任务标识。
    #   payload：精确资产、模型及账户范围。
    #   identity：已经验证的账户身份。
    # 输出：
    #   result：解析结果、内容绑定与本次模型调用记录。
    @app.post("/v1/threads/{thread_id}/interpret", dependencies=[local])
    def interpret_asset(
        thread_id: str,
        payload: AssetInterpretationRequest,
        identity: VerifiedIdentity = verified_identity,
    ) -> dict[str, object]:
        from .asset_interpretation import interpretation_source
        from .mission_service import _assert_model_connections_in_plugin_snapshot

        require_identity_match(payload, identity)
        try:
            thread = store.get_thread(thread_id)
            if thread["selected_model"] != payload.model_id:
                raise ValueError("PREPARATION_MODEL_MISMATCH")
            scope = MemoryOwnerScope(
                owner_account_id=identity.owner_account_id,
                tenant_id=identity.tenant_id,
                organization_id=identity.organization_id,
                source_edition=payload.source_edition,
            )
            mission_service.account_memory.bind_thread(scope, thread_id)
            source = interpretation_source(
                store, payload.kind, payload.asset_id, payload.content_sha256
            )
            if payload.model_grant.startswith("ddc_"):
                connection = custom_models.consume_grant(
                    payload.model_grant, thread_id, payload.model_id
                )
            else:
                if payload.gateway_base_url is None:
                    raise ValueError("MODEL_GATEWAY_REQUIRED")
                provider, _plugin_id, capability_id = plugin_manager.model_binding_for_model(
                    payload.model_id
                )
                connection = ModelConnection(
                    selection_id=payload.model_id,
                    provider=provider,
                    model_id=payload.model_id,
                    api_key=payload.model_grant,
                    base_url=_validate_gateway(str(payload.gateway_base_url), identity.issuer),
                    api_style="chat-completions",
                    capability_id=capability_id,
                    source="default",
                    supports_image_input=plugin_manager.model_supports_image_input(
                        payload.model_id
                    ),
                )
            _assert_model_connections_in_plugin_snapshot(
                plugin_manager.snapshot(thread_id=thread_id), connection, {}
            )
            result = mission_service.asset_interpretations.interpret(
                scope=scope.model_dump(mode="json"),
                source=source,
                connection=connection,
                locale=payload.locale,
                force=payload.force,
            )
            store.append_message(
                thread_id,
                role="assistant",
                kind="status",
                content=result["understanding"]["summary"],
                metadata={
                    "asset_interpretation": {
                        key: value for key, value in result.items() if key != "source"
                    },
                    "asset_id": payload.asset_id,
                    "content_sha256": payload.content_sha256,
                    "actuator_authority": False,
                },
            )
            return result
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="ASSET_INTERPRETATION_SOURCE_NOT_FOUND"
            ) from error
        except (ValueError, OSError, sqlite3.Error, ModelInvocationError) as error:
            code = str(error).split(":", 1)[0]
            if not code.startswith("ASSET_INTERPRETATION_"):
                code = "ASSET_INTERPRETATION_FAILED"
            raise HTTPException(status_code=409, detail=code) from error

    @app.post("/v1/threads/{thread_id}/prepare", dependencies=[local])
    def prepare(
        thread_id: str,
        payload: MissionPrepareRequest,
        identity_token: str | None = Header(default=None, alias="X-DroneDream-Identity-Token"),
        publishable_key: str | None = Header(
            default=None, alias="X-DroneDream-Supabase-Publishable-Key"
        ),
        identity: VerifiedIdentity = verified_identity,
    ) -> dict[str, object]:
        try:
            require_identity_match(payload, identity)
            projection_credentials, projection_configuration_issue = _projection_credentials(
                identity, identity_token, publishable_key
            )
            thread = store.get_thread(thread_id)
            if payload.model_id != thread["selected_model"]:
                raise ValueError("PREPARATION_MODEL_MISMATCH")
            if thread.get("locale") != payload.locale:
                thread = store.patch_thread(thread_id, {"locale": payload.locale})
            if payload.model_grant.startswith("ddc_"):
                connection = custom_models.consume_grant(
                    payload.model_grant, thread_id, payload.model_id
                )
            else:
                if payload.gateway_base_url is None:
                    raise ValueError("MODEL_GATEWAY_REQUIRED")
                provider, _plugin_id, capability_id = plugin_manager.model_binding_for_model(
                    payload.model_id
                )
                connection = ModelConnection(
                    selection_id=payload.model_id,
                    provider=provider,
                    model_id=payload.model_id,
                    api_key=payload.model_grant,
                    base_url=_validate_gateway(str(payload.gateway_base_url), identity.issuer),
                    api_style="chat-completions",
                    capability_id=capability_id,
                    source="default",
                    supports_image_input=plugin_manager.model_supports_image_input(
                        payload.model_id
                    ),
                )
            role_connections: dict[str, ModelConnection] = {}
            for role_model in payload.role_models:
                if role_model.role_port in role_connections:
                    raise ValueError("DUPLICATE_ROLE_MODEL_PORT")
                if role_model.model_grant.startswith("ddc_"):
                    role_connection = custom_models.consume_grant(
                        role_model.model_grant, thread_id, role_model.model_id
                    )
                else:
                    if role_model.gateway_base_url is None:
                        raise ValueError("MODEL_GATEWAY_REQUIRED")
                    role_provider, _plugin_id, role_capability_id = (
                        plugin_manager.model_binding_for_model(role_model.model_id)
                    )
                    role_connection = ModelConnection(
                        selection_id=role_model.model_id,
                        provider=role_provider,
                        model_id=role_model.model_id,
                        api_key=role_model.model_grant,
                        base_url=_validate_gateway(
                            str(role_model.gateway_base_url), identity.issuer
                        ),
                        api_style="chat-completions",
                        capability_id=role_capability_id,
                        source="default",
                        supports_image_input=plugin_manager.model_supports_image_input(
                            role_model.model_id
                        ),
                    )
                role_connections[role_model.role_port] = role_connection
            store.patch_thread(
                thread_id,
                {
                    "selected_model": payload.model_id,
                    "selected_map_id": payload.map_id,
                    "selected_map_content_sha256": payload.map_content_sha256,
                    "selected_vehicle_id": payload.vehicle_id,
                    "selected_vehicle_content_sha256": payload.vehicle_content_sha256,
                },
            )
            store.append_message(thread_id, role="user", kind="text", content=payload.message)
            result = mission_service.prepare(
                thread_id=thread_id,
                message=payload.message,
                map_id=payload.map_id,
                map_content_sha256=payload.map_content_sha256,
                vehicle_id=payload.vehicle_id,
                vehicle_content_sha256=payload.vehicle_content_sha256,
                connection=connection,
                role_connections=role_connections,
                locale=payload.locale,
                start_entity=payload.start_entity,
                attachment_ids=payload.attachment_ids,
                input_channel=payload.input_channel,
                input_metadata=payload.input_metadata,
                owner_account_id=identity.owner_account_id,
                tenant_id=identity.tenant_id,
                organization_id=identity.organization_id,
                source_edition=payload.source_edition,
                memory_projection_credentials=projection_credentials,
                memory_projection_configuration_issue=projection_configuration_issue,
            )
            notifications = result.get("notifications")
            if not isinstance(notifications, list) or not notifications:
                notifications = [
                    {
                        "kind": "plan",
                        "content": (
                            f"Plan ready: {result['goal']}"
                            if payload.locale == "en-US"
                            else f"计划已生成：{result['goal']}"
                        ),
                        "metadata": {},
                    }
                ]
            for notification in notifications:
                if not isinstance(notification, dict):
                    continue
                content = notification.get("content")
                kind = notification.get("kind")
                if not isinstance(content, str) or kind not in {"plan", "status"}:
                    continue
                metadata = notification.get("metadata")
                store.append_message(
                    thread_id,
                    role="assistant",
                    kind=kind,
                    content=content,
                    metadata=(
                        {**result, **metadata}
                        if kind == "plan" and isinstance(metadata, dict)
                        else metadata
                        if isinstance(metadata, dict)
                        else {}
                    ),
                )
            store.set_thread_state(thread_id, "awaiting_confirmation")
            return result
        except MissionClarificationRequired as error:
            # 追问不是计划；不进入 awaiting_confirmation，也不签发执行权限。
            store.set_thread_state(thread_id, "planning")
            text = ("请确认：" if payload.locale == "zh-CN" else "Please clarify: ") + "；".join(
                error.fields
            )
            store.append_message(
                thread_id,
                role="assistant",
                kind="text",
                content=text,
                metadata={"clarification_required": True, "actuator_authority": False},
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "MISSION_CLARIFICATION_REQUIRED",
                    "fields": error.fields,
                },
            ) from error
        except MissionPreparationBlocked as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ModelInvocationError as error:
            raise HTTPException(status_code=409, detail="MODEL_INVOCATION_FAILED") from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail="RESOURCE_NOT_FOUND") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/custom-models/discover", dependencies=[local])
    def discover_custom_models(payload: CustomModelDiscoverRequest) -> dict[str, object]:
        require_custom_model_connector()
        try:
            return custom_models.discover(base_url=payload.base_url, api_key=payload.api_key)
        except Exception as error:
            raise HTTPException(
                status_code=422, detail=f"CUSTOM_MODEL_DISCOVERY_FAILED:{type(error).__name__}"
            ) from error

    @app.post("/v1/custom-models", dependencies=[local], status_code=201)
    def create_custom_model(payload: CustomModelCreateRequest) -> dict[str, object]:
        require_custom_model_connector()
        try:
            return custom_models.create(**payload.model_dump())
        except (KeyError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/custom-models/{profile_id}/test", dependencies=[local])
    def test_custom_model(profile_id: str) -> dict[str, object]:
        require_custom_model_connector()
        try:
            return custom_models.test(profile_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="CUSTOM_MODEL_NOT_FOUND") from error
        except Exception as error:
            raise HTTPException(
                status_code=422, detail=f"CUSTOM_MODEL_TEST_FAILED:{type(error).__name__}"
            ) from error

    @app.post("/v1/custom-models/{profile_id}/grants", dependencies=[local])
    def issue_custom_model_grant(profile_id: str, thread_id: str) -> dict[str, object]:
        require_custom_model_connector()
        try:
            store.get_thread(thread_id)
            return custom_models.issue_grant(profile_id, thread_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="CUSTOM_MODEL_OR_THREAD_NOT_FOUND"
            ) from error

    @app.delete("/v1/custom-models/{profile_id}", dependencies=[local])
    def delete_custom_model(profile_id: str) -> dict[str, object]:
        try:
            custom_models.delete(profile_id)
            return {"deleted": True, "profile_id": profile_id}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="CUSTOM_MODEL_NOT_FOUND") from error

    @app.get("/v1/plugins", dependencies=[local])
    def plugins() -> list[dict[str, object]]:
        return plugin_manager.list_plugins()

    def harness_error(error: HarnessDesignServiceError) -> HTTPException:
        detail = str(error)
        if detail.startswith("HARNESS_REVISION_CONFLICT"):
            return HTTPException(status_code=409, detail=detail)
        if detail.endswith("NOT_FOUND"):
            return HTTPException(status_code=404, detail=detail)
        return HTTPException(status_code=422, detail=detail)

    @app.get("/v1/harness/catalog", dependencies=[local])
    def harness_catalog() -> dict[str, object]:
        return harness_design.catalog()

    @app.get("/v1/harness/profiles", dependencies=[local])
    def harness_profiles() -> list[dict[str, object]]:
        return harness_design.profiles()

    @app.get("/v1/harness/topologies/current", dependencies=[local])
    def current_harness_topology() -> dict[str, object]:
        return harness_design.current()

    @app.post("/v1/harness/topologies/validate", dependencies=[local])
    def validate_harness_topology(
        candidate: HarnessTopologyCandidate,
    ) -> dict[str, object]:
        return harness_design.validate(candidate).model_dump(mode="json")

    @app.patch("/v1/harness/topologies/current", dependencies=[local])
    def edit_harness_topology(operation: HarnessEditOperation) -> dict[str, object]:
        try:
            return harness_design.apply_operation(operation)
        except HarnessDesignServiceError as error:
            raise harness_error(error) from error

    @app.post("/v1/harness/topologies/undo", dependencies=[local])
    def undo_harness_topology(payload: HarnessRevisionActionRequest) -> dict[str, object]:
        try:
            return harness_design.undo(payload.base_revision)
        except HarnessDesignServiceError as error:
            raise harness_error(error) from error

    @app.post("/v1/harness/topologies/redo", dependencies=[local])
    def redo_harness_topology(payload: HarnessRevisionActionRequest) -> dict[str, object]:
        try:
            return harness_design.redo(payload.base_revision)
        except HarnessDesignServiceError as error:
            raise harness_error(error) from error

    @app.post("/v1/harness/topologies/dry-run", dependencies=[local])
    def dry_run_harness_topology(
        candidate: HarnessTopologyCandidate | None = None,
    ) -> dict[str, object]:
        return harness_design.dry_run(candidate)

    @app.get("/v1/harness/receipts", dependencies=[local])
    def harness_receipts(limit: int = 100) -> list[dict[str, object]]:
        return harness_design.receipts(limit=limit)

    @app.get("/v1/harness/topologies/{revision}", dependencies=[local])
    def harness_topology_revision(revision: int) -> dict[str, object]:
        try:
            return harness_design.get_revision(revision).model_dump(mode="json")
        except HarnessDesignServiceError as error:
            raise harness_error(error) from error

    @app.post("/v1/harness/topologies/{revision}/activate", dependencies=[local])
    def activate_harness_topology(revision: int) -> dict[str, object]:
        try:
            return harness_design.activate(revision)
        except HarnessDesignServiceError as error:
            raise harness_error(error) from error

    @app.get("/v1/plugin-governance", dependencies=[local])
    def plugin_governance() -> dict[str, object]:
        return {
            "policy": plugin_manager.governance_policy().model_dump(mode="json"),
            "decisions": store.list_plugin_governance_decisions(limit=200),
        }

    @app.put("/v1/plugin-governance", dependencies=[local])
    def replace_plugin_governance(policy: PluginGovernancePolicy) -> dict[str, object]:
        try:
            return plugin_manager.set_governance_policy(policy)
        except PluginManagerError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/v1/plugin-usage", dependencies=[local])
    def plugin_usage(limit: int = 200) -> list[dict[str, object]]:
        return store.list_plugin_usage(limit=limit)

    @app.get("/v1/plugin-marketplace", dependencies=[local])
    def plugin_marketplace_catalog() -> dict[str, object]:
        return plugin_marketplace.catalog()

    @app.put("/v1/plugin-marketplace/sources", dependencies=[local])
    def replace_plugin_marketplace_sources(
        sources: list[PluginMarketplaceSource],
    ) -> list[dict[str, object]]:
        try:
            return plugin_marketplace.replace_sources(sources)
        except PluginMarketplaceError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/plugin-marketplace/install", dependencies=[local])
    def install_marketplace_plugin(
        payload: PluginMarketplaceInstallRequest,
    ) -> dict[str, object]:
        try:
            return plugin_marketplace.install(**payload.model_dump())
        except PluginMarketplaceError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except PluginManagerError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/v1/plugins/{plugin_id}", dependencies=[local])
    def plugin(plugin_id: str) -> dict[str, object]:
        try:
            return plugin_manager.get_plugin(plugin_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="PLUGIN_NOT_FOUND") from error

    @app.get("/v1/plugins/{plugin_id}/panel", dependencies=[local])
    def plugin_panel(plugin_id: str, thread_id: str | None = None) -> dict[str, object]:
        try:
            data_sources: dict[str, object] = {"runtime": runtime_manager.status()}
            if thread_id:
                thread = store.get_thread(thread_id)
                mission_root = store.missions_root / thread_id
                evidence: list[dict[str, object]] = []
                if mission_root.is_dir():
                    for path in sorted(mission_root.glob("plan-*/*.json"), reverse=True)[:100]:
                        evidence.append(
                            {
                                "name": path.name,
                                "plan": path.parent.name,
                                "byte_size": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                        )
                data_sources.update(
                    {
                        "task": thread,
                        "evidence": {"items": evidence},
                    }
                )
            return plugin_manager.ui_panel_document(plugin_id, data_sources=data_sources)
        except (KeyError, PluginManagerError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/v1/plugins/import", dependencies=[local], status_code=201)
    async def import_plugin(bundle: Annotated[UploadFile, File()]) -> dict[str, object]:
        suffix = Path(bundle.filename or "plugin.zip").suffix
        descriptor, temporary_name = tempfile.mkstemp(prefix="dd-plugin-", suffix=suffix)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            await _stage_upload(
                bundle,
                temporary,
                maximum_bytes=512 * 1024 * 1024,
                too_large_detail="PLUGIN_UPLOAD_TOO_LARGE",
            )
            return plugin_manager.import_bundle(temporary)
        except PluginManagerError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            await anyio.Path(temporary).unlink(missing_ok=True)

    def plugin_action(plugin_id: str, action: str) -> dict[str, object]:
        try:
            if action == "enable":
                plugin = plugin_manager.get_plugin(plugin_id)
                placement = plugin.get("placement")
                if (
                    isinstance(placement, dict)
                    and placement.get("slot_id") == "harness.workflow-topology"
                ):
                    topology_id = str(
                        next(
                            capability["metadata"]["topology_id"]
                            for capability in plugin["capabilities"]
                            if capability["kind"] == "workflow-topology"
                        )
                    )
                    active_revision = harness_design.current()["current"]["revision"]
                    harness_design.apply_operation(
                        HarnessEditOperation(
                            client_operation_id=f"plugin-switch-{uuid4().hex[:24]}",
                            base_revision=int(active_revision),
                            operation="apply_template",
                            payload={"topology_id": topology_id},
                        )
                    )
                    return plugin_manager.get_plugin(plugin_id)
                return plugin_manager.enable(plugin_id)
            if action == "disable":
                return plugin_manager.disable(plugin_id)
            return plugin_manager.healthcheck(plugin_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="PLUGIN_NOT_FOUND") from error
        except PluginManagerError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except HarnessDesignServiceError as error:
            raise harness_error(error) from error

    @app.post("/v1/plugins/{plugin_id}/enable", dependencies=[local])
    def enable_plugin(plugin_id: str) -> dict[str, object]:
        return plugin_action(plugin_id, "enable")

    @app.post("/v1/plugins/{plugin_id}/trust-local-package", dependencies=[local])
    def trust_local_plugin_package(plugin_id: str) -> dict[str, object]:
        try:
            return plugin_manager.approve_local_package(plugin_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="PLUGIN_NOT_FOUND") from error
        except PluginManagerError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post(
        "/v1/plugins/{plugin_id}/versions/{version}/trust-local-package",
        dependencies=[local],
    )
    def trust_local_plugin_version(plugin_id: str, version: str) -> dict[str, object]:
        try:
            return plugin_manager.approve_local_version(plugin_id, version)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="PLUGIN_VERSION_NOT_FOUND") from error
        except PluginManagerError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/plugins/{plugin_id}/revoke-package", dependencies=[local])
    def revoke_plugin_package(plugin_id: str) -> dict[str, object]:
        try:
            return plugin_manager.revoke_package(plugin_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="PLUGIN_NOT_FOUND") from error
        except PluginManagerError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/plugin-publishers", dependencies=[local], status_code=201)
    def add_plugin_publisher(payload: TrustedPublisherRequest) -> dict[str, object]:
        try:
            return plugin_manager.add_trusted_publisher(**payload.model_dump())
        except PluginManagerError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/plugins/{plugin_id}/disable", dependencies=[local])
    def disable_plugin(plugin_id: str) -> dict[str, object]:
        return plugin_action(plugin_id, "disable")

    @app.post("/v1/plugins/{plugin_id}/healthcheck", dependencies=[local])
    def healthcheck_plugin(plugin_id: str) -> dict[str, object]:
        return plugin_action(plugin_id, "healthcheck")

    @app.post("/v1/plugins/{plugin_id}/apply-profile", dependencies=[local])
    def apply_plugin_profile(plugin_id: str) -> dict[str, object]:
        try:
            active_revision = harness_design.current()["current"]["revision"]
            result = harness_design.apply_operation(
                HarnessEditOperation(
                    client_operation_id=f"plugin-profile-{uuid4().hex[:24]}",
                    base_revision=int(active_revision),
                    operation="apply_profile",
                    payload={"profile_id": plugin_id},
                )
            )
            return {
                "profile": plugin_manager.get_plugin(plugin_id),
                "harness_revision": result["revision"],
                "receipt": result["receipt"],
            }
        except KeyError as error:
            raise HTTPException(status_code=404, detail="PLUGIN_NOT_FOUND") from error
        except PluginManagerError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except HarnessDesignServiceError as error:
            raise harness_error(error) from error

    @app.patch("/v1/plugins/{plugin_id}/configuration", dependencies=[local])
    def configure_plugin(plugin_id: str, payload: PluginConfigurationRequest) -> dict[str, object]:
        try:
            return plugin_manager.configure(plugin_id, payload.configuration)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="PLUGIN_NOT_FOUND") from error
        except (PluginManagerError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/plugins/{plugin_id}/rollback", dependencies=[local])
    def rollback_plugin(plugin_id: str, payload: PluginRollbackRequest) -> dict[str, object]:
        try:
            return plugin_manager.rollback(plugin_id, payload.version)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="PLUGIN_VERSION_NOT_FOUND") from error
        except PluginManagerError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/plugins/{plugin_id}/activate", dependencies=[local])
    def activate_plugin_version(
        plugin_id: str, payload: PluginRollbackRequest
    ) -> dict[str, object]:
        try:
            return plugin_manager.activate_version(plugin_id, payload.version)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="PLUGIN_VERSION_NOT_FOUND") from error
        except PluginManagerError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/plugins/{plugin_id}/promote", dependencies=[local])
    def promote_plugin_version(plugin_id: str, payload: PluginRollbackRequest) -> dict[str, object]:
        try:
            return plugin_manager.promote_version(plugin_id, payload.version)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="PLUGIN_VERSION_NOT_FOUND") from error
        except PluginManagerError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.delete("/v1/plugins/{plugin_id}", dependencies=[local])
    def uninstall_plugin(plugin_id: str) -> dict[str, object]:
        try:
            return plugin_manager.uninstall(plugin_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="PLUGIN_NOT_FOUND") from error
        except PluginManagerError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.patch("/v1/settings", dependencies=[local])
    def patch_settings(
        payload: SettingsPatch,
        identity_token: str | None = Header(default=None, alias="X-DroneDream-Identity-Token"),
        publishable_key: str | None = Header(
            default=None, alias="X-DroneDream-Supabase-Publishable-Key"
        ),
    ) -> dict[str, object]:
        updated = store.patch_settings(payload.model_dump(exclude_unset=True))
        if payload.memory_enabled is None:
            return updated
        if not identity_token:
            return {
                **updated,
                "memory_projection": MemoryProjectionSyncStatus(
                    status="configuration_missing",
                    issues=("MEMORY_PROJECTION_VERIFIED_IDENTITY_REQUIRED",),
                ).model_dump(mode="json"),
            }
        identity = authenticate_identity(identity_token)
        scope = MemoryOwnerScope(
            owner_account_id=identity.owner_account_id,
            tenant_id=identity.tenant_id,
            organization_id=identity.organization_id,
            source_edition="autonomy",
        )
        projection_issues: list[str] = []
        try:
            mission_service.memory_projection.queue_consent(scope, enabled=payload.memory_enabled)
        except (OSError, ValueError, sqlite3.Error) as error:
            projection_issues.append(f"MEMORY_PROJECTION_CONSENT_QUEUE_REJECTED:{error}"[:240])
        projection = sync_account_memory_projection(
            scope,
            identity,
            identity_token,
            publishable_key,
            local_memory_enabled=payload.memory_enabled,
            additional_issues=tuple(projection_issues),
        )
        return {**updated, "memory_projection": projection}

    @app.get("/v1/runtime/status", dependencies=[local])
    def runtime_status() -> dict[str, object]:
        return runtime_manager.status()

    @app.post("/v1/runtime/provision", dependencies=[local])
    def provision_runtime() -> dict[str, object]:
        try:
            return runtime_manager.provision()
        except RuntimeBridgeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/v1/runtime/setup", dependencies=[local])
    def runtime_setup_progress() -> dict[str, object]:
        return runtime_manager.setup_progress()

    @app.post("/v1/runtime/setup", dependencies=[local], status_code=202)
    def start_runtime_setup() -> dict[str, object]:
        return runtime_manager.start_setup()

    @app.post("/v1/threads/{thread_id}/execute", dependencies=[local], status_code=202)
    def execute_mission(
        thread_id: str,
        payload: MissionExecuteRequest,
        identity: VerifiedIdentity = verified_identity,
    ) -> dict[str, object]:
        try:
            require_identity_match(payload, identity)
            owner_scope = require_runtime_thread_scope(
                thread_id,
                identity,
                source_edition=payload.source_edition,
            )
            if payload.model_grant.startswith("ddc_"):
                connection = custom_models.consume_grant(
                    payload.model_grant, thread_id, payload.model_id
                )
            else:
                if payload.gateway_base_url is None:
                    raise ValueError("MODEL_GATEWAY_REQUIRED")
                provider, _plugin_id, capability_id = plugin_manager.model_binding_for_model(
                    payload.model_id
                )
                connection = ModelConnection(
                    selection_id=payload.model_id,
                    provider=provider,
                    model_id=payload.model_id,
                    api_key=payload.model_grant,
                    base_url=_validate_gateway(str(payload.gateway_base_url), identity.issuer),
                    api_style="chat-completions",
                    capability_id=capability_id,
                    source="default",
                    supports_image_input=plugin_manager.model_supports_image_input(
                        payload.model_id
                    ),
                )
            return runtime_manager.execute(
                thread_id=thread_id,
                connection=connection,
                plan_revision_id=payload.plan_revision_id,
                owner_scope=owner_scope,
            )
        except ValueError as error:
            if str(error) == "ACCOUNT_MEMORY_THREAD_OWNER_MISMATCH":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="EXECUTION_THREAD_OWNER_MISMATCH",
                ) from error
            raise HTTPException(status_code=409, detail=str(error)) from error
        except KeyError as error:
            detail = (
                "EXECUTION_THREAD_NOT_BOUND"
                if error.args and str(error.args[0]) == thread_id
                else str(error)
            )
            raise HTTPException(status_code=409, detail=detail) from error
        except RuntimeBridgeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/v1/threads/{thread_id}/execution-evidence", dependencies=[local])
    def execution_evidence(
        thread_id: str, identity: VerifiedIdentity = verified_identity
    ) -> dict[str, object]:
        try:
            require_runtime_thread_scope(thread_id, identity)
            return runtime_manager.execution_evidence(thread_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeBridgeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/v1/threads/{thread_id}/live-sources", dependencies=[local])
    def live_sources(
        thread_id: str, identity: VerifiedIdentity = verified_identity
    ) -> dict[str, object]:
        require_runtime_thread_scope(thread_id, identity)
        return runtime_manager.live_sources(thread_id)

    @app.get("/v1/threads/{thread_id}/live-frame", dependencies=[local])
    def live_frame(thread_id: str, identity: VerifiedIdentity = verified_identity) -> Response:
        require_runtime_thread_scope(thread_id, identity)
        path = runtime_manager.live_frame(thread_id)
        if path is None:
            raise HTTPException(status_code=404, detail="LIVE_FRAME_NOT_READY")
        try:
            body = path.read_bytes()
        except OSError as error:
            raise HTTPException(status_code=404, detail="LIVE_FRAME_NOT_READY") from error
        return Response(
            content=body,
            media_type="image/png",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/v1/threads/{thread_id}/live-telemetry", dependencies=[local])
    def live_telemetry(
        thread_id: str, identity: VerifiedIdentity = verified_identity
    ) -> dict[str, object]:
        require_runtime_thread_scope(thread_id, identity)
        value = runtime_manager.live_telemetry(thread_id)
        if value is None:
            raise HTTPException(status_code=404, detail="LIVE_TELEMETRY_NOT_READY")
        return value

    @app.post("/v1/threads/{thread_id}/runtime-message", dependencies=[local], status_code=202)
    def runtime_message(
        thread_id: str,
        payload: RuntimeMessageRequest,
        identity: VerifiedIdentity = verified_identity,
    ) -> dict[str, object]:
        try:
            require_runtime_thread_scope(thread_id, identity)
            return runtime_manager.submit_message(thread_id, payload.text)
        except (KeyError, RuntimeBridgeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post(
        "/v1/threads/{thread_id}/operator-takeover-grant",
        dependencies=[local],
        status_code=201,
    )
    def operator_takeover_grant(
        thread_id: str,
        payload: OperatorTakeoverGrantRequest,
        identity: VerifiedIdentity = verified_identity,
    ) -> dict[str, object]:
        try:
            require_runtime_thread_scope(thread_id, identity)
            if payload.operator_id != identity.owner_account_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="OPERATOR_IDENTITY_MISMATCH",
                )
            return runtime_manager.issue_takeover_grant(
                thread_id,
                message_id=payload.message_id,
                operator_id=payload.operator_id,
                duration_seconds=payload.duration_seconds,
            )
        except (KeyError, RuntimeBridgeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post(
        "/v1/threads/{thread_id}/operator-control",
        dependencies=[local],
        status_code=202,
    )
    def operator_control(
        thread_id: str,
        payload: OperatorControlRequest,
        identity: VerifiedIdentity = verified_identity,
    ) -> dict[str, object]:
        try:
            require_runtime_thread_scope(thread_id, identity)
            return runtime_manager.submit_operator_control(
                thread_id,
                operator_id=identity.owner_account_id,
                message_id=payload.message_id,
                grant_token=payload.grant_token,
                action=payload.action,
                north_mps=payload.north_mps,
                east_mps=payload.east_mps,
                down_mps=payload.down_mps,
                yaw_rate_dps=payload.yaw_rate_dps,
                duration_seconds=payload.duration_seconds,
            )
        except (KeyError, RuntimeBridgeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--plugin-isolator", type=Path)
    parser.add_argument("--account-memory-db", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.host != "127.0.0.1" or len(args.token) < 32:
        raise SystemExit(64)
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    store = AppStore(args.data_root)
    uvicorn.run(
        create_app(
            store=store,
            token=args.token,
            resource_root=args.resource_root,
            plugin_isolator_path=args.plugin_isolator,
            account_memory_path=args.account_memory_db or default_account_memory_path(),
        ),
        host=args.host,
        port=args.port,
        access_log=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
