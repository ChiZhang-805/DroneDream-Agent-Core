"""Bridge the desktop application to the real multi-call mission orchestrator."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from dronedream_agent_core.assets import load_map_catalog
from dronedream_agent_core.capability_broker import (
    CoreCapabilityBroker,
    CredentialResolver,
)
from dronedream_agent_core.context import ContextStore
from dronedream_agent_core.contracts import (
    AttachmentArtifact,
    EvidenceRecord,
    MapAsset,
    MissionRequest,
    VehicleAsset,
    ToolReceipt,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.evidence import EvidenceChain
from dronedream_agent_core.model_harness.boundary import (
    HARD_MAXIMUM_REPAIR_CYCLES,
    HarnessInputEnvelope,
    HarnessOutputEnvelope,
    autonomy_runtime_envelope,
    compile_autonomy_control_plane_receipt,
    compile_execution_authority,
    harness_input_sha256,
    selections_from_plugin_snapshot,
    validate_output_against_boundaries,
)
from dronedream_agent_core.model_harness.graph import HarnessTopology
from dronedream_agent_core.model_harness.memory import (
    AUTONOMY_MISSION_NAMESPACE,
    AccountMemoryStore,
    MemoryOwnerScope,
    PluginMemoryCandidate,
    empty_account_memory_model_context,
    validate_account_memory_model_context,
)
from dronedream_agent_core.model_harness.memory_projection import (
    AccountMemoryProjectionBridge,
    MemoryProjectionCredentials,
    MemoryProjectionSyncStatus,
)
from dronedream_agent_core.model_harness.model_port import (
    ProviderSettings,
    StructuredModelPort,
)
from dronedream_agent_core.orchestrator import MissionOrchestrator, PreparationConfig
from dronedream_agent_core.plugin_api import ToolEnvironment
from dronedream_agent_core.plugin_contracts import (
    CapabilityBrokerReceipt,
    PluginHookReceipt,
    PluginSnapshot,
)
from dronedream_agent_core.verification import verification_plan_matches_prepared_mission

from .asset_interpretation import (
    AssetInterpretationService,
    interpret_mission_assets,
    interpretation_source,
)
from .asset_runtime_resolver import (
    resolve_versioned_map,
    resolve_versioned_vehicle,
)
from .custom_models import ModelConnection
from .harness_design_service import HarnessDesignService
from .plugin_manager import PluginManager
from .storage import AppStore


# 功能：
#   限定平台模型网关的协议、路径和账户项目，避免把调用授权发送给其他项目。
# 输入：
#   value：申请模型授权时返回的网关地址。
#   identity_issuer：可选的已验签账户发行方，HTTP 路由必须提供。
# 输出：
#   gateway：通过检查并移除尾斜杠的网关地址。
def _validate_gateway(value: str, identity_issuer: str | None = None) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or not parsed.hostname
        or not parsed.hostname.endswith(".supabase.co")
        or parsed.port not in (None, 443)
        or parsed.path.rstrip("/") != "/functions/v1/model-gateway"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("INVALID_MODEL_GATEWAY")
    if identity_issuer is not None and parsed.hostname != urlparse(identity_issuer).hostname:
        raise ValueError("MODEL_GATEWAY_ACCOUNT_PROJECT_MISMATCH")
    return value.rstrip("/")


# 功能：
#   为通知插件提供独立的小型摘要，完整计划和证据留在任务产物中，不跨通知边界重复传输。
# 输入：
#   summary：桌面任务的完整结果。
# 输出：
#   notification_summary：只包含通知合同声明的标量字段。
def _plan_notification_summary(summary: dict[str, object]) -> dict[str, object]:
    fields = (
        "goal", "contract_id", "plan_revision_id", "plugin_snapshot_id", "locale",
        "plugin_catalog_sha256", "minimum_clearance_m", "model_calls",
        "planning_attempts", "target_entity", "return_entity",
    )
    notification_summary = {}
    for key in fields:
        if key not in summary:
            continue
        value = summary[key]
        if type(value) not in (str, int, float, bool, type(None)):
            raise ValueError("NOTIFICATION_SUMMARY_FIELD_INVALID")
        notification_summary[key] = value
    return notification_summary


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# 功能：
#   校验证据全链并保存完整工具回执索引，用有界引用承载长任务，绝不截断或丢弃记录。
# 输入：
#   output_dir：当前计划目录；evidence：完整证据链；tool_receipts：完整工具回执。
# 输出：
#   evidence_ids、tool_ids、index：证据链末端引用、工具引用和完整索引的文件绑定。
def _output_receipt_references(output_dir: Path, evidence: list[EvidenceRecord], tool_receipts: list[ToolReceipt]) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, object]]:
    EvidenceChain.verify(evidence)
    payload = {
        "schema_version": "dronedream.harness-receipt-index.v1",
        "evidence_record_ids": [item.record_sha256 for item in evidence],
        "tool_receipts": [item.model_dump(mode="json") for item in tool_receipts],
    }
    path = output_dir / "harness-receipt-index.json"
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)
    checksum = hashlib.sha256(raw).hexdigest()
    # 末端真实记录的摘要递归绑定全部前驱，而不是只保留任意一条日志。
    evidence_ids = (evidence[-1].record_sha256,) if evidence else ()
    tool_ids = tuple(item.call_id for item in tool_receipts)
    if len(tool_ids) > 128:
        tool_ids = (f"sha256:{checksum}",)
    index = {"path": path.name, "sha256": checksum, "evidence_count": len(evidence),
             "tool_count": len(tool_receipts), "evidence_reference": "verified-chain-head",
             "tool_reference": "individual" if len(tool_receipts) <= 128 else "complete-index"}
    return evidence_ids, tool_ids, index


# 功能：
#   将已准备的任务投影到界面协议，保留插件动作和实际规划预算，不赋予飞行权限。
# 输入：
#   prepared：已验证任务；input_metadata：请求绑定；public_*：公开资产标识与版本；maximum_planning_rounds：配置上限。
# 输出：
#   artifact：供界面展示与摘要绑定的计划对象。
def _public_planner_artifact(
    *,
    prepared,
    input_metadata: dict[str, object],
    public_map_id: str,
    public_map_version: int,
    public_aircraft_id: str,
    public_aircraft_version: int,
    maximum_planning_rounds: int,
) -> dict[str, object]:
    if not 1 <= prepared.planning_attempts <= maximum_planning_rounds <= 5:
        raise ValueError("PUBLIC_PLANNER_ATTEMPT_BUDGET_INVALID")
    context_sha256 = str(input_metadata.get("public_harness_context_sha256", ""))
    if len(context_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in context_sha256
    ):
        raise ValueError("PUBLIC_HARNESS_CONTEXT_SHA256_INVALID")
    nodes = [
        {
            "node_id": node.task_id,
            "action": node.action,
            "target": node.target_node,
            "depends_on": list(node.depends_on),
            "success_evidence": list(node.success_evidence),
        }
        for node in prepared.task_graph.nodes
    ]
    tool_receipts = [receipt.model_dump(mode="json") for receipt in prepared.tool_receipts]
    return {
        "schema_version": "dronedream.autonomy.planner-response.v1",
        "status": "draft",
        "goal": prepared.intent.goal,
        "asset_bindings": {
            "aircraft_id": public_aircraft_id,
            "aircraft_version": public_aircraft_version,
            "map_id": public_map_id,
            "map_version": public_map_version,
            "context_sha256": context_sha256,
        },
        "grounded_entities": [
            {"role": "start", "entity": prepared.intent.start_entity},
            {"role": "target", "entity": prepared.intent.target_entity},
            {"role": "return", "entity": prepared.intent.return_entity},
        ],
        "task_graph": {"nodes": nodes},
        "tool_requests": [
            {
                "call_id": receipt.call_id,
                "tool_id": receipt.tool_id,
                "plugin_id": receipt.plugin_id,
            }
            for receipt in prepared.tool_receipts
        ],
        "tool_receipts": tool_receipts,
        "assumptions": list(prepared.intent.assumptions),
        "blockers": [],
        "repair": {
            "attempt": prepared.planning_attempts,
            "max_attempts": maximum_planning_rounds,
            "repeated_plan_hashes": 0,
            "stop_reason": None,
        },
        "safety_policy": {
            "actuator_authority": False,
            "may_relax_constraints": False,
            "execution_requires_deterministic_validation": True,
        },
    }


def _distance_between(left, right) -> float:
    return math.sqrt((right.x - left.x) ** 2 + (right.y - left.y) ** 2 + (right.z - left.z) ** 2)


def _prepared_plan_gates(prepared) -> dict[str, bool]:
    """Project only deterministic preparation facts into public feasibility.

    Historical edge-flight labels improve candidate ranking, but a newly generated
    metric route cannot possess them before its first run.  Continuous clearance and
    immutable artifact bindings are the actual pre-execution gates.
    """

    verification_plan_bound = True
    if prepared.verification_plan is not None:
        verification_plan_bound = bool(
            prepared.runtime_checkpoints is not None
            and prepared.runtime_actions is not None
            and verification_plan_matches_prepared_mission(
                verification_plan=prepared.verification_plan,
                contract=prepared.contract,
                domain_actions=prepared.domain_actions,
                task_graph=prepared.task_graph,
                semantic_plan=prepared.semantic_plan,
                flight_plan=prepared.plan,
                execution_route=prepared.execution_route,
                route_clearance=prepared.route_clearance,
                px4_track=prepared.px4_track,
                runtime_checkpoints=prepared.runtime_checkpoints,
                runtime_actions=prepared.runtime_actions,
            )
        )
    return {
        "intent_review_accepted": prepared.intent_critique.accepted,
        "plan_review_accepted": prepared.plan_critique.accepted,
        "continuous_clearance_accepted": prepared.route_clearance.accepted,
        "clearance_route_hash_matches": (
            prepared.route_clearance.route_sha256 == sha256_json(prepared.execution_route)
        ),
        "clearance_semantic_hash_matches": (
            prepared.route_clearance.semantic_sha256 == prepared.contract.map_semantic_sha256
        ),
        "verification_plan_bound": verification_plan_bound,
    }


def _public_mission_plan(
    *,
    prepared,
    graph: MapAsset,
    vehicle: VehicleAsset,
    map_content_sha256: str,
    vehicle_content_sha256: str,
) -> dict[str, object]:
    """Expose only values proven by the hash-bound Core preparation.

    The public shell must not re-create arbitrary imported assets with its
    bundled School Map/My Drone defaults.  This projection therefore carries
    the exact Core asset hashes and leaves metrics that the imported vehicle
    contract does not prove as ``None``.
    """

    action_labels = {action.action_id: action.label for action in prepared.domain_actions.actions}
    segment_by_task: dict[str, list[object]] = {}
    estimated_duration_s = 0.0
    for segment in prepared.plan.segments:
        segment_by_task.setdefault(segment.task_id, []).append(segment)
        distance_m = sum(
            _distance_between(left.position_m, right.position_m)
            for left, right in zip(segment.path, segment.path[1:], strict=False)
        )
        estimated_duration_s += distance_m / segment.speed_limit_mps
    estimated_duration_s += max(0, len(prepared.px4_track.points) - 1) * float(
        prepared.px4_track.waypoint_hold_seconds
    )

    route_positions = prepared.execution_route.positions_m
    vertical_travel_m = sum(
        abs(right.z - left.z)
        for left, right in zip(route_positions, route_positions[1:], strict=False)
    )

    def risk_for(fallback: str) -> str:
        if fallback in {"abort", "land"}:
            return "high"
        if fallback in {"hold", "return"}:
            return "medium"
        return "low"

    def executor_for(action: str) -> str:
        if action in {"takeoff", "traverse", "navigate", "return", "land"}:
            return "px4_bridge"
        if "payload" in action or action in {"pickup", "dropoff", "release"}:
            return "payload_controller"
        if "inspect" in action or "detect" in action or "track" in action:
            return "perception"
        return "mission_executive"

    graph_nodes = []
    steps = []
    for order, node in enumerate(prepared.task_graph.nodes, start=1):
        task_segments = segment_by_task.get(node.task_id, [])
        timeout_s = max(
            10.0,
            sum(
                sum(
                    _distance_between(left.position_m, right.position_m)
                    for left, right in zip(segment.path, segment.path[1:], strict=False)
                )
                / segment.speed_limit_mps
                for segment in task_segments
            )
            * 2.0
            + 5.0,
        )
        graph_nodes.append(
            {
                "task_id": node.task_id,
                "label": action_labels.get(node.action, node.action.replace("_", " ")),
                "status": "ready" if not node.depends_on else "pending",
                "depends_on": list(node.depends_on),
                "executor": executor_for(node.action),
                "risk": risk_for(node.fallback),
                "max_retries": node.max_retries,
                "timeout_s": round(min(timeout_s, 3_600.0), 3),
                "fallback": "hold" if node.fallback == "return" else node.fallback,
                "expected_output": "; ".join(node.success_evidence),
                "completion_evidence": list(node.success_evidence),
                "inserted_by": "compiler",
            }
        )
        steps.append(
            {
                "order": order,
                "action": node.action,
                "label": action_labels.get(node.action, node.action.replace("_", " ")),
                "payload_delta_kg": 0.0,
            }
        )

    assumptions = [
        {
            "code": f"intent.assumption.{index}",
            "severity": "info",
            "message": assumption,
        }
        for index, assumption in enumerate(prepared.intent.assumptions, start=1)
    ]
    sensor_names = {item.casefold() for item in vehicle.sensors}
    has_visual_sensor = bool(
        sensor_names.intersection({"rgb", "depth", "stereo", "thermal", "vio"})
    )
    perception_mode = "fusion" if has_visual_sensor else "map"
    prepared_sha256 = _canonical_sha256(prepared.model_dump(mode="json"))
    preparation_gates = _prepared_plan_gates(prepared)
    feasible = all(preparation_gates.values())
    return {
        "schema_version": "dronedream.agent-core-mission-plan.v1",
        "source": "agent-core",
        "contract_id": prepared.contract.contract_id,
        "prepared_mission_sha256": prepared_sha256,
        "scene_id": graph.asset_id,
        "scene_name": graph.name,
        "feasible": feasible,
        "readiness": "simulation_ready",
        "can_execute": bool(prepared.status == "awaiting_confirmation" and feasible),
        "requires_confirmation": True,
        "perception_mode": perception_mode,
        "asset_bindings": {
            "map_asset_id": prepared.contract.map_asset_id,
            "map_content_sha256": map_content_sha256,
            "vehicle_asset_id": prepared.contract.vehicle_asset_id,
            "vehicle_content_sha256": vehicle_content_sha256,
        },
        "steps": steps,
        "route_positions_m": [point.model_dump(mode="json") for point in route_positions],
        "task_graph": {
            "schema_version": "dronedream.autonomy.task-graph.v1",
            "revision": prepared.task_graph.revision,
            "nodes": graph_nodes,
            "active_node_ids": [],
            "change_reason": "agent-core-preparation",
        },
        "issues": assumptions,
        "verification": {
            "preparation_gates": preparation_gates,
            "historical_edge_flight_labels_complete": (
                prepared.execution_route.all_edges_flight_verified
            ),
            "requirements": (
                [
                    requirement.model_dump(mode="json")
                    for requirement in prepared.verification_plan.requirements
                ]
                if prepared.verification_plan is not None
                else []
            ),
        },
        "metrics": {
            "route_length_m": prepared.execution_route.route_length_m,
            "vertical_travel_m": vertical_travel_m,
            "estimated_duration_s": estimated_duration_s,
            "minimum_clearance_m": prepared.route_clearance.minimum_clearance_m,
            "launch_mass_kg": vehicle.dry_mass_kg,
            "post_pickup_mass_kg": None,
            "post_pickup_thrust_to_weight": None,
            "braking_distance_m": None,
        },
        "immutable_safety_rules": list(prepared.contract.immutable_safety_rules),
    }


def _assert_model_connections_in_plugin_snapshot(
    snapshot: PluginSnapshot,
    primary: ModelConnection,
    role_connections: dict[str, ModelConnection],
) -> None:
    """Bind every model port to the immutable plugin snapshot used by this plan.

    Managed models are already resolved through ``model_binding_for_model`` at
    the HTTP boundary. Custom grants, however, can outlive a UI toggle for a few
    minutes. Rechecking here makes the ``models.providers`` slot operational:
    disabling the custom connector prevents a previously issued grant from
    silently bypassing the selected Harness snapshot.
    """

    provider_models: dict[str, set[str]] = {}
    custom_connector_enabled = False
    for entry in snapshot.plugins:
        manifest = entry.manifest
        if manifest.placement.slot_id != "models.providers":
            continue
        for capability in manifest.capabilities:
            if capability.kind != "model-provider":
                continue
            provider = capability.metadata.get("provider")
            if provider == "custom":
                custom_connector_enabled = True
                continue
            models = capability.metadata.get("models")
            if not isinstance(provider, str) or not isinstance(models, list):
                continue
            allowed = provider_models.setdefault(provider, set())
            allowed.update(
                str(model["id"])
                for model in models
                if isinstance(model, dict) and isinstance(model.get("id"), str) and model["id"]
            )

    for port_name, connection in {"primary": primary, **role_connections}.items():
        if connection.source == "custom":
            if not custom_connector_enabled:
                raise ValueError(f"CUSTOM_MODEL_PROVIDER_PLUGIN_DISABLED:{port_name}")
            continue
        if connection.model_id not in provider_models.get(connection.provider, set()):
            raise ValueError(
                f"MODEL_NOT_BOUND_TO_PLUGIN_SNAPSHOT:{port_name}:{connection.model_id}"
            )


class MissionService:
    def __init__(
        self,
        store: AppStore,
        plugin_manager: PluginManager,
        credential_resolver: CredentialResolver | None = None,
        harness_design: HarnessDesignService | None = None,
        account_memory_path: Path | None = None,
    ) -> None:
        self.store = store
        self.plugin_manager = plugin_manager
        self.credential_resolver = credential_resolver
        self.harness_design = harness_design
        self.asset_interpretations = AssetInterpretationService(store)
        from .airspace_service import AirspaceService
        self.airspace = AirspaceService(store)
        self.account_memory = AccountMemoryStore(
            account_memory_path or (store.root / "account-memory.sqlite3")
        )
        self.memory_projection = AccountMemoryProjectionBridge(self.account_memory)

    def _decode_attachments(
        self,
        *,
        thread_id: str,
        attachment_ids: list[str],
        extension_registry,
    ) -> tuple[list[AttachmentArtifact], list[PluginHookReceipt]]:
        artifacts: list[AttachmentArtifact] = []
        all_receipts: list[PluginHookReceipt] = []
        for attachment_id in attachment_ids:
            record = self.store.get_attachment(attachment_id, thread_id)
            source = Path(str(record["local_path"])).resolve()
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            outputs, receipts = extension_registry.invoke_multiple(
                "input.attachment-decoders",
                "decode_attachment",
                path=str(source),
                attachment_id=attachment_id,
                display_name=str(record["display_name"]),
                content_type=str(record["content_type"]),
                size_bytes=int(record["byte_size"]),
                source_sha256=source_sha256,
            )
            all_receipts.extend(receipts)
            candidates = [
                (output, receipt)
                for output, receipt in zip(outputs, receipts, strict=True)
                if isinstance(output, dict) and output.get("accepted") is True
            ]
            if not candidates:
                raise ValueError(f"ATTACHMENT_DECODER_NOT_FOUND:{attachment_id}")
            decoded, receipt = max(
                candidates,
                key=lambda item: int(item[0].get("priority", 0)),
            )
            artifacts.append(
                AttachmentArtifact(
                    attachment_id=attachment_id,
                    display_name=str(record["display_name"]),
                    content_type=str(record["content_type"]),
                    size_bytes=int(record["byte_size"]),
                    source_sha256=source_sha256,
                    decoder_plugin_id=receipt.plugin_id,
                    decoded_kind=str(decoded["decoded_kind"]),  # type: ignore[arg-type]
                    text=(str(decoded["text"]) if decoded.get("text") is not None else None),
                    structured_data=(
                        dict(decoded["structured_data"])
                        if isinstance(decoded.get("structured_data"), dict)
                        else {}
                    ),
                    model_input=(
                        dict(decoded["model_input"])
                        if isinstance(decoded.get("model_input"), dict)
                        else {}
                    ),
                    issue_codes=[str(item) for item in decoded.get("issue_codes", [])],
                )
            )
        return artifacts, all_receipts

    def prepare(
        self,
        *,
        thread_id: str,
        message: str,
        map_id: str,
        map_content_sha256: str | None,
        vehicle_id: str,
        vehicle_content_sha256: str | None,
        connection: ModelConnection,
        role_connections: dict[str, ModelConnection] | None = None,
        locale: str,
        start_entity: str,
        attachment_ids: list[str],
        input_channel: str = "text",
        input_metadata: dict[str, object] | None = None,
        owner_account_id: str,
        tenant_id: str | None = None,
        organization_id: str | None = None,
        source_edition: str | None = None,
        memory_projection_credentials: MemoryProjectionCredentials | None = None,
        memory_projection_configuration_issue: str | None = None,
    ) -> dict[str, object]:
        owner_scope = MemoryOwnerScope(
            owner_account_id=owner_account_id,
            tenant_id=tenant_id,
            organization_id=organization_id,
            source_edition=source_edition,
        )
        self.account_memory.bind_thread(owner_scope, thread_id)
        app_settings = self.store.get_settings()
        memory_enabled = bool(app_settings.get("memory_enabled", True))
        persisted_task_context = memory_enabled and bool(
            app_settings.get("remember_task_preferences", True)
        )
        effective_input_metadata = dict(input_metadata or {})
        # The caller cannot inject a memory envelope. Retrieval is owner-bound
        # here and validated again after every plugin boundary in the Core.
        effective_input_metadata.pop("long_term_memory", None)
        effective_input_metadata.pop("asset_understanding", None)
        memory_issues: list[str] = []
        try:
            projection_status = self.memory_projection.sync(
                owner_scope,
                memory_projection_credentials,
                local_memory_enabled=memory_enabled,
                operation_limit=6,
                pull_limit=32,
                time_budget_seconds=3.0,
            )
        except (OSError, sqlite3.Error, ValueError) as error:
            projection_status = MemoryProjectionSyncStatus(
                status="conflict",
                pending_operations=self.memory_projection.pending_count(owner_scope),
                issues=(f"MEMORY_PROJECTION_SYNC_REJECTED:{error}"[:240],),
            )
        memory_issues.extend(projection_status.issues)
        if memory_projection_configuration_issue:
            memory_issues.append(memory_projection_configuration_issue[:240])
        if persisted_task_context:
            try:
                long_term_memory = self.account_memory.model_context(
                    owner_scope,
                    query=message,
                    limit=8,
                    token_budget=1_200,
                )
            except (OSError, sqlite3.Error, ValueError) as error:
                memory_issues.append(f"ACCOUNT_MEMORY_RETRIEVAL_SKIPPED:{error}")
                long_term_memory = empty_account_memory_model_context()
        else:
            long_term_memory = empty_account_memory_model_context()
        effective_input_metadata["long_term_memory"] = long_term_memory
        effective_input_metadata["memory_policy"] = {
            "namespace": AUTONOMY_MISSION_NAMESPACE,
            "owner_bound": True,
            "persisted_task_context": persisted_task_context,
            "remember_asset_choices": memory_enabled
            and bool(app_settings.get("remember_asset_choices", True)),
            "safety_boundaries_are_non_relaxable": True,
            "cloud_projection": projection_status.model_dump(mode="json"),
        }
        harness_revision_binding: dict[str, object] = {}
        harness_topology_override: HarnessTopology | None = None
        if self.harness_design is not None:
            frozen_harness = self.harness_design.freeze_active_for_task()
            harness_revision_binding = {
                key: value for key, value in frozen_harness.items() if key != "topology"
            }
            harness_topology_override = HarnessTopology.model_validate(frozen_harness["topology"])
            effective_input_metadata["harness_revision"] = harness_revision_binding
        if map_content_sha256 is None or vehicle_content_sha256 is None:
            raise ValueError("ASSET_VERSION_PAIR_REQUIRED")
        map_selection = resolve_versioned_map(
            self.store.get_asset_version(map_id, map_content_sha256)
        )
        vehicle_selection = resolve_versioned_vehicle(
            self.store.get_asset_version(vehicle_id, vehicle_content_sha256)
        )
        pair_qualification = self.store.qualified_asset_pair(
            map_asset_id=map_selection.asset_id,
            map_content_sha256=map_selection.content_sha256,
            vehicle_asset_id=vehicle_selection.asset_id,
            vehicle_content_sha256=vehicle_selection.content_sha256,
        )
        if pair_qualification is None:
            raise ValueError("ASSET_PAIR_QUALIFICATION_REQUIRED")
        pair_binding = self.store.verified_asset_pair_execution_binding(pair_qualification)
        effective_input_metadata["asset_versions"] = {
            "map": {
                "asset_id": map_selection.asset_id,
                "content_sha256": map_selection.content_sha256,
            },
            "vehicle": {
                "asset_id": vehicle_selection.asset_id,
                "content_sha256": vehicle_selection.content_sha256,
            },
        }
        effective_input_metadata["asset_pair_qualification"] = pair_binding.model_dump(mode="json")
        provider = connection.provider
        map_root = map_selection.root
        vehicle_root = vehicle_selection.root
        graph_path = map_selection.graph
        semantic_path = map_selection.semantic
        vehicle_sdf = vehicle_selection.vehicle_sdf
        graph = MapAsset.model_validate_json(graph_path.read_text(encoding="utf-8"))
        vehicle = VehicleAsset.model_validate_json(
            vehicle_selection.vehicle_metadata.read_text(encoding="utf-8")
        )
        from .airspace_service import AirspaceRequest
        preferred_space = self.airspace.field(AirspaceRequest(
            map_asset_id=map_selection.asset_id,
            map_content_sha256=map_selection.content_sha256,
            vehicle_asset_id=vehicle_selection.asset_id,
            vehicle_content_sha256=vehicle_selection.content_sha256,
        ))
        preferred_context = {
            "airspace_sha256": preferred_space.sha256,
            "binding": preferred_space.binding,
            "authority": "preference-only; live safety and payload checks required",
            "named_node_height_samples": [
                {"node_id": node.node_id, "space": preferred_space.context(
                    (node.position_m.x, node.position_m.y, node.position_m.z))}
                for node in graph.nodes[:96]
            ],
            "node_samples_truncated": len(graph.nodes) > 96,
        }
        effective_input_metadata["preferred_airspace"] = preferred_context
        if start_entity == "__auto__":
            launch_node_ids = {node.node_id for node in graph.nodes if node.semantic == "launch"}
            launch_aliases = [
                alias
                for alias, node_id in graph.named_entities.items()
                if node_id in launch_node_ids
            ]
            if not launch_aliases:
                raise ValueError("MAP_LAUNCH_ENTITY_REQUIRED")
            launch_aliases.sort(
                key=lambda value: (
                    0
                    if any(term in value.casefold() for term in ("launch", "start", "起飞", "起点"))
                    else 1,
                    len(value),
                    value,
                )
            )
            start_entity = launch_aliases[0]
        catalog = load_map_catalog(semantic_path, qualified_graph=graph)
        mission_root = self.store.missions_root / thread_id
        mission_root.mkdir(parents=True, exist_ok=True)
        revision = len(list(mission_root.glob("plan-*"))) + 1
        output_dir = mission_root / f"plan-{revision:04d}"
        attachment_root = self.store.attachments_root / thread_id
        attachment_root.mkdir(parents=True, exist_ok=True)
        broker_receipts: list[CapabilityBrokerReceipt] = []
        broker_factory = CoreCapabilityBroker(
            read_roots={
                "map": map_root,
                "vehicle": vehicle_root,
                "attachments": attachment_root,
            },
            write_roots={
                "output": output_dir,
                "staging": self.store.plugin_staging_root / thread_id / f"plan-{revision:04d}",
            },
            credential_resolver=self.credential_resolver,
            receipt_sink=broker_receipts.append,
        )
        plugin_snapshot = self.plugin_manager.snapshot(thread_id=thread_id)
        _assert_model_connections_in_plugin_snapshot(
            plugin_snapshot,
            connection,
            role_connections or {},
        )
        _policy_entry, _policy_manifest, policy_capability = (
            self.plugin_manager.capability_for_slot(plugin_snapshot, "planning.workflow-policy")
        )
        policy = policy_capability.metadata
        _retry_entry, _retry_manifest, retry_capability = self.plugin_manager.capability_for_slot(
            plugin_snapshot, "harness.retry-policy"
        )
        _timeout_entry, _timeout_manifest, timeout_capability = (
            self.plugin_manager.capability_for_slot(plugin_snapshot, "harness.timeout-policy")
        )
        _budget_entry, _budget_manifest, budget_capability = (
            self.plugin_manager.capability_for_slot(plugin_snapshot, "harness.budget-policy")
        )
        retry_policy = retry_capability.metadata
        timeout_policy = timeout_capability.metadata
        budget_policy = budget_capability.metadata
        maximum_intent_rounds = int(policy.get("max_intent_rounds", 3))
        maximum_planning_rounds = int(policy.get("max_planning_rounds", 5))
        maximum_model_calls = int(budget_policy.get("maximum_model_calls", 48))
        maximum_repair_cycles = maximum_intent_rounds + maximum_planning_rounds - 2
        if not 0 <= maximum_repair_cycles <= HARD_MAXIMUM_REPAIR_CYCLES:
            raise ValueError("HARNESS_REPAIR_BUDGET_EXCEEDS_FIXED_KERNEL_LIMIT")
        selected_plugins = selections_from_plugin_snapshot(plugin_snapshot)
        control_plane_receipt = compile_autonomy_control_plane_receipt(
            selected_plugins,
            effective_maximum_model_calls=maximum_model_calls,
            effective_maximum_repair_cycles=maximum_repair_cycles,
        )
        runtime_envelope = autonomy_runtime_envelope(
            owner_scope,
            control_plane_receipt,
            plugin_snapshot,
            selected_maximum_model_calls=maximum_model_calls,
            maximum_intent_rounds=maximum_intent_rounds,
            maximum_planning_rounds=maximum_planning_rounds,
        )
        maximum_model_calls = runtime_envelope.effective_maximum_model_calls
        effective_input_metadata["model_harness_control_plane"] = control_plane_receipt.model_dump(
            mode="json"
        )
        effective_input_metadata["model_harness_runtime"] = runtime_envelope.model_dump(mode="json")
        tool_registry = self.plugin_manager.build_tool_registry(
            environment=ToolEnvironment(
                map_graph=graph,
                semantic_path=semantic_path,
                vehicle_diameter_m=vehicle.body_radius_m * 2,
                vehicle_height_m=vehicle.body_height_m,
                waypoint_hold_seconds=0.4,
                vehicle=vehicle,
                broker_factory=broker_factory,
                planning_phase="initial",
            ),
            snapshot=plugin_snapshot,
        )
        extension_registry = self.plugin_manager.build_extension_registry(
            snapshot=plugin_snapshot,
            broker_factory=broker_factory,
        )
        ranked_memory, account_memory_retrieval_receipts = extension_registry.invoke_pipeline(
            "memory.account-retrieval",
            "rank_account_memory",
            long_term_memory,
            locale=locale,
            input_channel=input_channel,
        )
        # Plugins may filter/rank already-governed items, but cannot expand the
        # fixed schema/budget or introduce identity and execution authority.
        effective_input_metadata["long_term_memory"] = validate_account_memory_model_context(
            ranked_memory
        )
        long_term_memory = effective_input_metadata["long_term_memory"]
        request_id = f"request-{uuid4().hex}"
        input_envelope = HarnessInputEnvelope(
            request_id=request_id,
            task_id=thread_id,
            thread_id=thread_id,
            owner_binding_sha256=runtime_envelope.owner_binding_sha256,
            tenant_binding_sha256=runtime_envelope.tenant_binding_sha256,
            source_edition=runtime_envelope.source_edition,
            control_plane_selection_sha256=control_plane_receipt.selection_sha256,
            current_request={
                "message": message,
                "locale": locale,
                "input_channel": input_channel,
                "start_entity": start_entity,
                "attachment_ids": list(attachment_ids),
                "map_asset_id": map_selection.asset_id,
                "map_content_sha256": map_selection.content_sha256,
                "vehicle_asset_id": vehicle_selection.asset_id,
                "vehicle_content_sha256": vehicle_selection.content_sha256,
                "model_selection_id": connection.selection_id,
            },
            session_context={
                "memory_policy": effective_input_metadata["memory_policy"],
                "asset_versions": effective_input_metadata["asset_versions"],
                "asset_pair_qualification": effective_input_metadata["asset_pair_qualification"],
                "preferred_airspace": preferred_context,
                "harness_revision": harness_revision_binding,
            },
            memory_record_ids=tuple(
                str(item["memory_key"])
                for item in long_term_memory["items"]
                if isinstance(item, dict) and isinstance(item.get("memory_key"), str)
            ),
        )
        # This exact canonical envelope travels through MissionRequest and is
        # therefore part of the real orchestrator/model path, not just a report.
        effective_input_metadata["model_harness_input"] = input_envelope.model_dump(mode="json")
        attachments, attachment_receipts = self._decode_attachments(
            thread_id=thread_id,
            attachment_ids=attachment_ids,
            extension_registry=extension_registry,
        )
        context = ContextStore(self.store.root / "mission-context.sqlite3")
        resources = ExitStack()
        resources.callback(context.close)

        def build_port(model_connection: ModelConnection) -> StructuredModelPort:
            settings = ProviderSettings(
                name=model_connection.provider,
                model=model_connection.model_id,
                api_key_env="DRONEDREAM_MODEL_CREDENTIAL",
                base_url=model_connection.base_url,
                api_style=model_connection.api_style,
                supports_image_input=model_connection.supports_image_input,
            )
            port = StructuredModelPort(
                model_connection.provider,
                max_attempts=int(retry_policy.get("provider_attempts", 3)),
                timeout_seconds=float(timeout_policy.get("model_seconds", 180.0)),
                settings=settings,
                api_key=model_connection.api_key,
            )
            resources.callback(port.close)
            return port

        try:
            primary = build_port(connection)
            critic = build_port((role_connections or {}).get("critic", connection))
            model_ports = {"primary": primary, "critic": critic}
            for role_port, role_connection in (role_connections or {}).items():
                if role_port not in model_ports:
                    model_ports[role_port] = build_port(role_connection)
            # 解析属于准备阶段：缓存命中不调用模型，新调用占用同一任务的总预算。
            interpretation_result = interpret_mission_assets(
                service=self.asset_interpretations,
                scope=owner_scope.model_dump(mode="json"),
                sources=(
                    interpretation_source(
                        self.store, "map", map_selection.asset_id, map_selection.content_sha256
                    ),
                    interpretation_source(
                        self.store,
                        "vehicle",
                        vehicle_selection.asset_id,
                        vehicle_selection.content_sha256,
                    ),
                ),
                connection=connection,
                locale=locale,
                port=primary,
                maximum_model_calls=maximum_model_calls,
            )
            interpretations = interpretation_result["interpretations"]
            interpretation_calls = interpretation_result["model_calls"]
            remaining_model_calls = interpretation_result["remaining_model_calls"]
            input_envelope = HarnessInputEnvelope.model_validate(
                {
                    **input_envelope.model_dump(mode="json"),
                    "session_context": {
                        **input_envelope.session_context,
                        "asset_understanding": interpretations,
                    },
                }
            )
            effective_input_metadata["model_harness_input"] = input_envelope.model_dump(mode="json")
            orchestrator = MissionOrchestrator(
                config=PreparationConfig(
                    provider=provider,  # type: ignore[arg-type]
                    critic_provider=provider,  # type: ignore[arg-type]
                    vehicle_diameter_m=vehicle.body_radius_m * 2,
                    vehicle_height_m=vehicle.body_height_m,
                    max_intent_rounds=maximum_intent_rounds,
                    max_planning_rounds=maximum_planning_rounds,
                    plugin_router_rounds=int(policy.get("plugin_router_rounds", 2)),
                    maximum_plugin_calls=int(policy.get("maximum_plugin_calls", 8)),
                    intent_reviews_per_round=int(policy.get("intent_reviews_per_round", 1)),
                    plan_reviews_per_round=int(policy.get("plan_reviews_per_round", 1)),
                    maximum_model_calls=remaining_model_calls,
                    maximum_optional_tool_calls=int(budget_policy.get("maximum_tool_calls", 16)),
                    model_timeout_seconds=float(timeout_policy.get("model_seconds", 180.0)),
                    persisted_task_context=persisted_task_context,
                ),
                map_catalog=catalog,
                map_graph=graph,
                semantic_path=semantic_path,
                vehicle_sdf=vehicle_sdf,
                vehicle_asset_id=vehicle.asset_id,
                vehicle=vehicle,
                context_store=context,
                primary_port=primary,
                critic_port=critic,
                model_ports=model_ports,
                tool_registry=tool_registry,
                extension_registry=extension_registry,
                plugin_snapshot=plugin_snapshot,
                initial_hook_receipts=[
                    *attachment_receipts,
                    *account_memory_retrieval_receipts,
                ],
                harness_topology_override=harness_topology_override,
                harness_revision_binding=harness_revision_binding,
            )
            resources.callback(orchestrator.close)
            prepared = orchestrator.prepare(
                MissionRequest(
                    conversation_id=thread_id,
                    message=message,
                    start_entity=start_entity,
                    locale=locale,  # type: ignore[arg-type]
                    attachments=attachments,
                    input_channel=input_channel,  # type: ignore[arg-type]
                    input_metadata=effective_input_metadata,
                ),
                output_dir,
            )
            binding = context.lifecycle.binding(thread_id)
        finally:
            # 解析失败、构造失败和准备成功走同一条释放链；外部传入编排器的模型端口由本层关闭。
            resources.close()
        consolidated_memory_keys: list[str] = []
        pending_memory_keys: list[str] = []
        account_memory_candidate_ids: list[str] = []
        account_memory_candidate_receipts: list[PluginHookReceipt] = []
        if persisted_task_context:
            candidates: list[tuple[str, str, dict[str, object], float, int, str]] = [
                (
                    "summary",
                    "summary.latest_mission",
                    {
                        "target_entity": prepared.intent.target_entity,
                        "return_entity": prepared.intent.return_entity,
                        "payload_action": prepared.intent.payload_action,
                        "locale": locale,
                    },
                    0.85,
                    90,
                    "model_inference",
                ),
                (
                    "preference",
                    "preference.task",
                    {"locale": locale, "input_channel": input_channel},
                    0.80,
                    365,
                    "model_inference",
                ),
            ]
            if prepared.intent.constraints:
                candidates.append(
                    (
                        "constraint",
                        "constraint.mission",
                        {"constraints": list(prepared.intent.constraints)},
                        0.90,
                        365,
                        "model_inference",
                    )
                )
            plugin_candidate_outputs, account_memory_candidate_receipts = (
                extension_registry.invoke_multiple(
                    "memory.candidate-extraction",
                    "extract_account_memory_candidates",
                    intent=prepared.intent.model_dump(mode="json"),
                    locale=locale,
                    input_channel=input_channel,
                )
            )
            candidate_keys = {item[1] for item in candidates}
            for output in plugin_candidate_outputs:
                raw_candidates = output if isinstance(output, list) else [output]
                for raw_candidate in raw_candidates:
                    if len(candidates) >= 16:
                        memory_issues.append("ACCOUNT_MEMORY_PLUGIN_CANDIDATE_LIMIT_REACHED")
                        break
                    try:
                        candidate = PluginMemoryCandidate.model_validate(raw_candidate)
                    except ValueError as error:
                        memory_issues.append(f"ACCOUNT_MEMORY_PLUGIN_CANDIDATE_REJECTED:{error}")
                        continue
                    if candidate.memory_key in candidate_keys:
                        memory_issues.append(
                            f"ACCOUNT_MEMORY_PLUGIN_CANDIDATE_DUPLICATE:{candidate.memory_key}"
                        )
                        continue
                    candidates.append(
                        (
                            candidate.kind,
                            candidate.memory_key,
                            candidate.payload,
                            candidate.confidence,
                            candidate.ttl_days,
                            "plugin_inference",
                        )
                    )
                    candidate_keys.add(candidate.memory_key)
            if bool(app_settings.get("remember_asset_choices", True)):
                candidates.append(
                    (
                        "preference",
                        "preference.assets",
                        {
                            "map_asset_id": map_selection.asset_id,
                            "vehicle_asset_id": vehicle_selection.asset_id,
                        },
                        0.80,
                        365,
                        "model_inference",
                    )
                )
            for (
                kind,
                memory_key,
                payload,
                confidence,
                ttl_days,
                source_kind,
            ) in candidates:
                try:
                    candidate_result = self.account_memory.record_candidate(
                        owner_scope,
                        kind=kind,  # type: ignore[arg-type]
                        memory_key=memory_key,
                        payload=payload,
                        source_conversation_id=thread_id,
                        confidence=confidence,
                        ttl_days=ttl_days,
                        source_kind=source_kind,  # type: ignore[arg-type]
                    )
                    account_memory_candidate_ids.append(candidate_result.candidate_id)
                    try:
                        self.memory_projection.queue_candidate(
                            owner_scope, candidate_result.candidate_id
                        )
                    except (KeyError, OSError, sqlite3.Error, ValueError) as error:
                        memory_issues.append(
                            f"MEMORY_PROJECTION_CANDIDATE_QUEUE_REJECTED:{memory_key}:{error}"[:240]
                        )
                    if candidate_result.promoted:
                        consolidated_memory_keys.append(memory_key)
                    else:
                        pending_memory_keys.append(memory_key)
                        if candidate_result.conflict_with_active:
                            memory_issues.append(f"ACCOUNT_MEMORY_CANDIDATE_CONFLICT:{memory_key}")
                except (sqlite3.Error, ValueError) as error:
                    # Unsafe or instruction-shaped content is never persisted;
                    # mission preparation remains available without that item.
                    memory_issues.append(f"ACCOUNT_MEMORY_CANDIDATE_REJECTED:{memory_key}:{error}")
            try:
                removed = self.account_memory.apply_retention(
                    owner_scope, maximum_entries=128, maximum_age_days=365
                )
            except (OSError, sqlite3.Error, ValueError) as error:
                removed = 0
                memory_issues.append(f"ACCOUNT_MEMORY_RETENTION_SKIPPED:{error}")
        else:
            removed = 0
        projection_pending = self.memory_projection.pending_count(owner_scope)
        if projection_pending != projection_status.pending_operations:
            projected_status = projection_status.status
            if projected_status == "synced" and projection_pending:
                projected_status = "pending"
            projection_status = projection_status.model_copy(
                update={
                    "status": projected_status,
                    "pending_operations": projection_pending,
                }
            )
        broker_receipt_path = output_dir / "capability-broker-receipts.json"
        broker_receipt_payload = [item.model_dump(mode="json") for item in broker_receipts]
        broker_receipt_path.write_text(
            json.dumps(broker_receipt_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        intent_call_count = sum(
            1 for record in prepared.model_calls if record.role == "intent_parser"
        )
        repair_cycle_count = max(0, intent_call_count - 1) + max(0, prepared.planning_attempts - 1)
        evidence_ids, tool_ids, receipt_index = _output_receipt_references(
            output_dir, prepared.evidence, prepared.tool_receipts)
        output_envelope = HarnessOutputEnvelope(
            request_id=input_envelope.request_id,
            task_id=input_envelope.task_id,
            control_plane_selection_sha256=control_plane_receipt.selection_sha256,
            input_envelope_sha256=harness_input_sha256(input_envelope),
            status="validated_proposal",
            structured_result={
                "status": prepared.status,
                "contract_id": prepared.contract.contract_id,
                "mission_id": binding.thread.mission_id,
                "plan_revision_id": binding.plan_revision.plan_revision_id,
                "receipt_index": receipt_index,
            },
            model_call_count=prepared.model_attempt_count + len(interpretation_calls),
            repair_cycle_count=repair_cycle_count,
            tool_receipt_ids=tool_ids,
            validation_receipt_ids=(prepared.contract.contract_id,),
            evidence_receipt_ids=evidence_ids,
            memory_candidate_ids=tuple(account_memory_candidate_ids[:32]),
        )
        validate_output_against_boundaries(
            control_plane_receipt,
            runtime_envelope,
            output_envelope,
            input_envelope=input_envelope,
        )
        execution_authority = compile_execution_authority(
            thread_id=thread_id,
            plan_revision_id=binding.plan_revision.plan_revision_id,
            contract_id=prepared.contract.contract_id,
            prepared_mission_sha256=sha256_json(prepared),
            runtime=runtime_envelope,
        )
        self.store.issue_execution_authority(execution_authority)
        (output_dir / "model-harness-execution-authority.json").write_text(
            execution_authority.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "locale": locale,
            "thread_id": thread_id,
            "mission_id": binding.thread.mission_id,
            "plan_revision_id": binding.plan_revision.plan_revision_id,
            "status": prepared.status,
            "contract_id": prepared.contract.contract_id,
            "goal": prepared.intent.goal,
            "target_entity": prepared.intent.target_entity,
            "return_entity": prepared.intent.return_entity,
            "planning_attempts": prepared.planning_attempts,
            "model_calls": prepared.model_attempt_count + len(interpretation_calls),
            "successful_model_responses": len(prepared.model_calls) + len(interpretation_calls),
            "asset_interpretations": interpretations,
            "asset_interpretation_calls": interpretation_calls,
            "model_selection_id": connection.selection_id,
            "model_id": connection.model_id,
            "model_source": connection.source,
            "role_models": {
                role_port: {
                    "selection_id": role_connection.selection_id,
                    "model_id": role_connection.model_id,
                    "provider": role_connection.provider,
                    "source": role_connection.source,
                }
                for role_port, role_connection in (role_connections or {}).items()
            },
            "plugin_snapshot_id": prepared.plugin_snapshot.snapshot_id,
            "plugin_catalog_sha256": prepared.plugin_snapshot.catalog_sha256,
            "execution_authority": execution_authority.model_dump(mode="json"),
            "route_nodes": prepared.execution_route.node_ids,
            "minimum_clearance_m": prepared.route_clearance.minimum_clearance_m,
            "mission_plan": _public_mission_plan(
                prepared=prepared,
                graph=graph,
                vehicle=vehicle,
                map_content_sha256=map_selection.content_sha256,
                vehicle_content_sha256=vehicle_selection.content_sha256,
            ),
            "output_dir": str(output_dir),
            "capability_broker_receipts": len(broker_receipts),
            "capability_broker_receipts_sha256": hashlib.sha256(
                broker_receipt_path.read_bytes()
            ).hexdigest(),
            "model_harness_control_plane": {
                "receipt": control_plane_receipt.model_dump(mode="json"),
                "runtime_envelope": runtime_envelope.model_dump(mode="json"),
                "input_envelope": input_envelope.model_dump(mode="json"),
                "output_envelope": output_envelope.model_dump(mode="json"),
            },
            "account_memory": {
                "namespace": AUTONOMY_MISSION_NAMESPACE,
                "owner_bound": True,
                "retrieved_items": len(long_term_memory["items"]),
                "retrieval_mode": "owner_bound_semantic_account_shared",
                "consolidated_keys": consolidated_memory_keys,
                "pending_keys": pending_memory_keys,
                "retention_removed": removed,
                "issues": memory_issues,
                "cloud_projection": projection_status.model_dump(mode="json"),
                "authority_reuse_allowed": False,
                "retrieval_plugin_receipts": [
                    receipt.model_dump(mode="json") for receipt in account_memory_retrieval_receipts
                ],
                "candidate_plugin_receipts": [
                    receipt.model_dump(mode="json") for receipt in account_memory_candidate_receipts
                ],
            },
        }
        public_bindings = effective_input_metadata.get("public_asset_bindings")
        if isinstance(public_bindings, dict):
            planner_artifact = _public_planner_artifact(
                prepared=prepared,
                input_metadata=effective_input_metadata,
                public_map_id=str(public_bindings.get("map_id", "")),
                public_map_version=int(public_bindings.get("map_version", 0)),
                public_aircraft_id=str(public_bindings.get("aircraft_id", "")),
                public_aircraft_version=int(public_bindings.get("aircraft_version", 0)),
                maximum_planning_rounds=maximum_planning_rounds,
            )
            artifact_bindings = planner_artifact["asset_bindings"]
            if not isinstance(artifact_bindings, dict) or (
                not artifact_bindings["map_id"]
                or artifact_bindings["map_version"] < 1
                or not artifact_bindings["aircraft_id"]
                or artifact_bindings["aircraft_version"] < 1
            ):
                raise ValueError("PUBLIC_ASSET_BINDINGS_INVALID")
            summary["integration_artifact"] = planner_artifact
            summary["integration_artifact_sha256"] = _canonical_sha256(planner_artifact)
            # 传递被摘要绑定的原始字节，防止 JavaScript 把 1.0、指数和键序重编码。
            summary["integration_artifact_canonical_json"] = json.dumps(
                planner_artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
        notification_outputs, notification_receipts = extension_registry.invoke_multiple(
            "notifications.plan-ready",
            "render_plan_notification",
            summary=_plan_notification_summary(summary),
        )
        notifications = [
            output
            for output in notification_outputs
            if isinstance(output, dict)
            and output.get("channel") == "task-timeline"
            and output.get("kind") in {"plan", "status"}
            and isinstance(output.get("content"), str)
        ]
        summary["notifications"] = notifications
        summary["notification_plugin_receipts"] = [
            receipt.model_dump(mode="json") for receipt in notification_receipts
        ]
        (output_dir / "desktop-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary
