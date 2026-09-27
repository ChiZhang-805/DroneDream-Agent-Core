"""Confirmed execution of a prepared mission in the real PX4/Gazebo/ROS stack."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from itertools import islice
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .assets import load_map_catalog
from .checkpointing import CheckpointCoordinator, checkpoint_contract_for
from .context import ContextStore
from .contracts import (
    CompletionAssessment,
    GraphRoute,
    MapAsset,
    MissionLifecycleBinding,
    ModelCallRecord,
    PreparedMission,
    Px4GazeboGates,
    Px4GazeboRunEvidence,
    Px4Track,
    RouteClearanceReport,
    RuntimeActionExecutionReceipt,
    RuntimeCheckpointContract,
    RuntimeCheckpointDecision,
    RuntimeCheckpointRequest,
    RuntimeControlSession,
    RuntimeInterruptionDecision,
    RuntimeReplacementTrack,
    SimulationWorkflowResult,
    VehicleAsset,
)
from .evidence import EvidenceChain
from .extensions import ExtensionExecutionError
from .gazebo_adapter import run_px4_gazebo_track
from .hashing import sha256_json
from .model_harness.boundary import ModelHarnessExecutionAuthority
from .model_harness.model_port import ProviderName, StructuredModelPort
from .plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file
from .prompts import COMPLETION_VERIFIER
from .runtime_control_io import publish_runtime_json, read_runtime_object
from .runtime_interrupt import (
    RuntimeInterruptionCoordinator,
    close_runtime_control_session,
    create_runtime_control_session,
)
from .runtime_plugins import (
    all_required_gates_passed,
    append_hook_receipts,
    augment_runtime_prompt,
    require_plugin_acceptance,
    runtime_extension_registry,
    validate_runtime_model_output,
)
from .runtime_revision import replacement_adoption_gates
from .verification import verification_plan_matches_prepared_mission


class PreparedMissionBindingError(RuntimeError):
    """The confirmed package no longer matches its structured artifacts or assets."""


COMPLETION_VERIFIER_INPUT_MAX_BYTES = 96 * 1024
_COMPLETION_COLLECTION_PREVIEW_ITEMS = 32
_EXECUTION_JSON_MAX_BYTES = 64 * 1024 * 1024
_ContractT = TypeVar("_ContractT", bound=BaseModel)


# 功能：
#   有界读取并严格解析执行合同，拒绝链接、重复 JSON 键、类型强转及读取期间的文件替换。
# 输入：
#   path：明确的制品路径；contract_type：期望合同类型。
# 输出：
#   artifact：重新验证的独立合同。
def _read_contract(path: Path, contract_type: type[_ContractT]) -> _ContractT:
    raw = decode_json(
        read_plugin_file(path, limit=_EXECUTION_JSON_MAX_BYTES),
        limit=_EXECUTION_JSON_MAX_BYTES,
        node_limit=1_000_000,
    )
    artifact = contract_type.model_validate_json(
        encode_json(raw, limit=_EXECUTION_JSON_MAX_BYTES, node_limit=1_000_000), strict=True
    )
    return artifact


# 功能：
#   重新验证内部模型并冻结调用边界，保留历史证据未显式设置字段的摘要语义。
# 输入：
#   value：内部合同对象。
# 输出：
#   artifact：与原对象分离的同类型合同。
def _owned_contract(value: _ContractT) -> _ContractT:
    artifact = type(value).model_validate(
        value.model_dump(mode="python", exclude_unset=True), strict=True
    )
    return artifact


# 功能：
#   限量枚举运行证据，存在超量或链接目录时拒绝验收，不静默忽略尾部。
# 输入：
#   directory：证据目录；pattern：由代码指定的文件模式。
# 输出：
#   paths：按名称排序的有限文件路径。
def _evidence_paths(directory: Path, pattern: str) -> list[Path]:
    check_plain_plugin_path(directory)
    paths = list(islice(directory.glob(pattern), 4097))
    if len(paths) > 4096:
        raise PreparedMissionBindingError("EXECUTION_EVIDENCE_COUNT_EXCEEDED")
    paths.sort()
    return paths


# 功能：
#   规范编码有限 JSON，用于完成模型输入预算；不把 NaN 或自定义对象变成有效证据。
# 输入：
#   value：结构化 JSON 值。
# 输出：
#   payload：排序后的 UTF-8 字节。
def _canonical_json_bytes(value: Any) -> bytes:
    encode_json(value, limit=_EXECUTION_JSON_MAX_BYTES, node_limit=1_000_000)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return payload


# 功能：
#   保留全量制品清单的数量、字节数与摘要，仅截断展示预览。
# 输入：
#   items：真实文件绑定清单。
# 输出：
#   summary：完整集合统计及有限预览。
def _artifact_collection_summary(items: list[Any]) -> dict[str, Any]:
    normalized = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in items
    ]
    if any(
        not isinstance(item, dict)
        or type(item.get("size_bytes")) is not int
        or item["size_bytes"] < 0
        for item in normalized
    ):
        raise ValueError("COMPLETION_ARTIFACT_SIZE_INVALID")
    preview = [
        {
            "sha256": item.get("sha256"),
            "size_bytes": item.get("size_bytes"),
        }
        if isinstance(item, dict)
        else item
        for item in normalized[:_COMPLETION_COLLECTION_PREVIEW_ITEMS]
    ]
    summary = {
        "count": len(normalized),
        "total_size_bytes": sum(item["size_bytes"] for item in normalized),
        "collection_sha256": sha256_json(normalized),
        "preview": preview,
        "preview_truncated": len(normalized) > len(preview),
    }
    return summary


# 功能：
#   为云端完成核验保留实际门控和测量，大型图像/日志清单通过完整摘要绑定。
# 输入：
#   runtime：本轮原始运行证据。
# 输出：
#   evidence：模型可读的紧凑运行证据。
def _runtime_evidence_for_completion(runtime: Px4GazeboRunEvidence) -> dict[str, Any]:
    runtime = _owned_contract(runtime)
    artifacts = runtime.artifacts.model_dump(mode="json")
    px4_ulogs = artifacts.pop("px4_ulogs", [])
    visual_frames = artifacts.pop("model_navigation_visual_frames", [])
    batching = artifacts.get("static_render_batching")
    if isinstance(batching, dict) and isinstance(batching.get("batches"), list):
        # 网格来源映射随地图大小增长，不是完成门控。仅摘要这份清单，
        # 物理不变性、失败原因和资格标记等其余字段仍逐项保留。
        batches = batching["batches"]
        batching["batches"] = {
            "count": len(batches),
            "collection_sha256": sha256_json(batches),
            "preview": [
                {"batch_sha256": sha256_json(item)}
                for item in batches[:_COMPLETION_COLLECTION_PREVIEW_ITEMS]
            ],
            "preview_truncated": len(batches) > _COMPLETION_COLLECTION_PREVIEW_ITEMS,
        }
    evidence = {
        "schema_version": runtime.schema_version,
        "status": runtime.status,
        "world": runtime.world,
        "vehicle": runtime.vehicle,
        "runtime_evidence_sha256": sha256_json(runtime),
        "gates": runtime.gates.model_dump(mode="json", exclude_unset=True),
        "measurements": runtime.measurements.model_dump(mode="json", exclude_none=True),
        "artifact_bindings": artifacts,
        "artifact_collections": {
            "px4_ulogs": _artifact_collection_summary(px4_ulogs),
            "model_navigation_visual_frames": _artifact_collection_summary(visual_frames),
        },
    }
    return evidence


# 功能：
#   压缩验证要求的预览，不丢失完整要求集合的摘要与总数。
# 输入：
#   prepared：本轮确认的任务包。
# 输出：
#   dumped：验证计划摘要；没有计划时为 None。
def _verification_plan_for_completion(prepared: PreparedMission) -> dict[str, Any] | None:
    plan = prepared.verification_plan
    if plan is None:
        return None
    dumped = plan.model_dump(mode="json")
    requirements = dumped.pop("requirements")
    dumped["requirements"] = {
        "count": len(requirements),
        "requirements_sha256": sha256_json(requirements),
        "items": requirements[:_COMPLETION_COLLECTION_PREVIEW_ITEMS],
        "items_truncated": len(requirements) > _COMPLETION_COLLECTION_PREVIEW_ITEMS,
    }
    return dumped


# 功能：
#   1. 组合任务、绑定门控、实际动作/检查点和飞行证据供云端独立核验。
#   2. 压缩大清单后计算准确字节数，仍超预算则拒绝调用，不删去关键失败证据。
# 输入：
#   prepared、route、clearance、track：确认包及计划制品；runtime、offboard_timing：运行反馈。
#   binding_gates、normalized_runtime_evaluations：固定校验与插件评估。
#   checkpoint_decisions、runtime_action_receipts、runtime_interruption_decisions：实际运行记录。
#   deterministic_success：代码层全部门控的联合结果。
# 输出：
#   artifact：有字节预算的完成核验输入。
def _completion_verifier_input(
    *,
    prepared: PreparedMission,
    route: GraphRoute,
    clearance: RouteClearanceReport,
    track: Px4Track,
    runtime: Px4GazeboRunEvidence,
    offboard_timing: dict[str, Any],
    binding_gates: dict[str, bool],
    normalized_runtime_evaluations: list[Any],
    checkpoint_decisions: list[RuntimeCheckpointDecision],
    runtime_action_receipts: list[RuntimeActionExecutionReceipt],
    runtime_interruption_decisions: list[RuntimeInterruptionDecision],
    deterministic_success: bool,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "input_contract": {
            "schema_version": "dronedream.completion-verifier-input.v1",
            "maximum_canonical_json_bytes": COMPLETION_VERIFIER_INPUT_MAX_BYTES,
            "unbounded_artifact_lists_represented_by_count_and_sha256": True,
        },
        "mission_contract": prepared.contract.model_dump(mode="json"),
        "execution_authorization": {
            "contract_confirmation_verified": True,
            "confirmed_contract_id": prepared.contract.contract_id,
            "pre_confirmation_constraints_satisfied": [
                constraint
                for constraint in prepared.contract.constraints
                if constraint in {"plan_only", "do_not_execute"}
            ],
            "semantics": (
                "The plan was displayed without execution; this run began only after "
                "the caller confirmed the exact immutable contract ID."
            ),
        },
        "hash_domains": {
            "canonical_json": {
                "prepared_mission_sha256": _prepared_mission_sha256(prepared),
                "route_sha256": sha256_json(route),
                "track_sha256": sha256_json(track),
                "clearance_sha256": sha256_json(clearance),
            },
            "runtime_file_bytes": {
                "route_sha256": runtime.artifacts.route_sha256,
                "track_sha256": runtime.artifacts.track_sha256,
                "clearance_sha256": runtime.artifacts.clearance_sha256,
            },
        },
        "binding_gates": binding_gates,
        "verification_plan": _verification_plan_for_completion(prepared),
        "plugin_runtime_evaluations": normalized_runtime_evaluations,
        "checkpoint_decisions": [
            {
                "request_sha256": decision.request_sha256,
                "action": decision.assessment.action,
                "issue_codes": decision.assessment.issue_codes,
                "continue_authorized": decision.continue_authorized,
                "model_output_sha256": decision.model_call.output_sha256,
            }
            for decision in checkpoint_decisions
        ],
        "runtime_interruption_decisions": [
            {
                "message_sha256": decision.message_sha256,
                "hold_ack_sha256": decision.hold_ack_sha256,
                "message_kind": decision.classification.message_kind,
                "requested_action": decision.classification.requested_action,
                "authorized_action": decision.authorized_action,
                "authorization_gates": decision.authorization_gates,
                "model_output_sha256": decision.model_call.output_sha256,
            }
            for decision in runtime_interruption_decisions
        ],
        "runtime_action_receipts": [
            receipt.model_dump(mode="json") for receipt in runtime_action_receipts
        ],
        "runtime_evidence": _runtime_evidence_for_completion(runtime),
        "offboard_timing": {
            "status": offboard_timing.get("status"),
            "cleanup": offboard_timing.get("cleanup"),
            "failure": offboard_timing.get("failure"),
            "track_end_t": offboard_timing.get("track_end_t"),
            "land_confirmed_t": offboard_timing.get("land_confirmed_t"),
        },
        "deterministic_success": deterministic_success,
    }
    artifact_size = len(_canonical_json_bytes(artifact))
    if artifact_size > COMPLETION_VERIFIER_INPUT_MAX_BYTES:
        # Plugin output is independently hash-bound by receipts and can be large.
        # Preserve its identity and count instead of allowing it to crowd out
        # safety gates or measurements from the completion decision.
        evaluations = artifact["plugin_runtime_evaluations"]
        artifact["plugin_runtime_evaluations"] = {
            "count": len(evaluations) if isinstance(evaluations, list) else 1,
            "evaluations_sha256": sha256_json(evaluations),
            "items_omitted_for_input_bound": True,
        }
        artifact_size = len(_canonical_json_bytes(artifact))
    artifact["input_contract"]["canonical_json_bytes"] = artifact_size
    while True:
        final_size = len(_canonical_json_bytes(artifact))
        if final_size == artifact["input_contract"]["canonical_json_bytes"]:
            break
        artifact["input_contract"]["canonical_json_bytes"] = final_size
    if final_size > COMPLETION_VERIFIER_INPUT_MAX_BYTES:
        raise PreparedMissionBindingError(
            "COMPLETION_VERIFIER_INPUT_EXCEEDS_BOUND:"
            f"{final_size}>{COMPLETION_VERIFIER_INPUT_MAX_BYTES}"
        )
    return artifact


# 功能：
#   只对已签发任务中实际出现的字段计算摘要，新增兼容默认值不能改变旧证据身份。
# 输入：
#   prepared：确认的任务包。
# 输出：
#   digest：规范 JSON 摘要。
def _prepared_mission_sha256(prepared: PreparedMission) -> str:
    model_dump = getattr(prepared, "model_dump", None)
    if not callable(model_dump):
        # Retain support for deliberately minimal test doubles used to exercise
        # lifecycle ordering without constructing the multi-megabyte contract.
        return sha256_json(prepared)
    digest = sha256_json(model_dump(mode="json", exclude_unset=True))
    return digest


# 功能：
#   要求必备运行门控齐全且本轮显式启用门控全为真，不把空集合或字符串当成功。
# 输入：
#   runtime：携带运行门控的证据。
# 输出：
#   accepted：门控联合结果。
def _active_runtime_gates_passed(runtime: Px4GazeboRunEvidence) -> bool:
    try:
        raw = runtime.gates.model_dump(mode="python", exclude_unset=True)
        Px4GazeboGates.model_validate(raw, strict=True)
    except (AttributeError, TypeError, ValueError):
        return False
    accepted = all_required_gates_passed(raw)
    return accepted


# 功能：
#   逐个核对已采纳替换的执行身份、完整制品与连续序号，残缺记录不能静默退回旧计划。
# 输入：
#   run_dir：当前运行目录；prepared：可选的已确认原任务包。
# 输出：
#   adopted：按采纳顺序排列的替换合同。
def _adopted_runtime_replacements(
    run_dir: Path, prepared: PreparedMission | None = None
) -> list[RuntimeReplacementTrack]:
    control_dir = run_dir / "runtime-control"
    adopted: list[RuntimeReplacementTrack] = []
    paths = _evidence_paths(control_dir / "adoptions", "*.json")
    if not paths:
        return adopted
    session = _read_contract(control_dir / "session.json", RuntimeControlSession)
    if prepared is not None and (
        session.contract_id != prepared.contract.contract_id
        or session.conversation_id != prepared.contract.conversation_id
        or session.prepared_mission_sha256 != _prepared_mission_sha256(prepared)
    ):
        raise PreparedMissionBindingError("RUNTIME_REPLACEMENT_SESSION_MISMATCH")
    for adoption_path in paths:
        replacement_path = control_dir / "replacements" / adoption_path.name
        if not replacement_path.is_file():
            raise PreparedMissionBindingError("RUNTIME_REPLACEMENT_ARTIFACT_MISSING")
        adoption = read_runtime_object(adoption_path)
        replacement = _read_contract(replacement_path, RuntimeReplacementTrack)
        gates = replacement_adoption_gates(
            adoption=adoption, replacement=replacement, execution_id=session.execution_id
        )
        gates["filename"] = adoption_path.name == f"{replacement.message_id}.json"
        if not all(gates.values()):
            failed = ",".join(name for name, accepted in gates.items() if not accepted)
            raise PreparedMissionBindingError(f"RUNTIME_REPLACEMENT_ADOPTION_INVALID:{failed}")
        adopted.append(replacement)
    adopted.sort(key=lambda item: item.replacement_sequence)
    if [item.replacement_sequence for item in adopted] != list(range(1, len(adopted) + 1)):
        raise PreparedMissionBindingError("RUNTIME_REPLACEMENT_SEQUENCE_INVALID")
    return adopted


# 功能：
#   合并原计划及已采纳修订的动作定义，核对每条实际回执并确定仍需完成的步骤。
# 输入：
#   prepared：原确认包；run_dir：本轮动作及替换证据目录。
# 输出：
#   result：按时间排序的回执列表和有效必做步骤集合。
def _load_runtime_action_receipts(
    prepared: PreparedMission, run_dir: Path
) -> tuple[list[RuntimeActionExecutionReceipt], set[str]]:
    replacements = _adopted_runtime_replacements(run_dir, prepared)
    active_track_sha256 = sha256_json(prepared.px4_track)
    for replacement in replacements:
        if replacement.prior_track_sha256 != active_track_sha256:
            raise PreparedMissionBindingError("RUNTIME_REPLACEMENT_TRACK_CHAIN_INVALID")
        active_track_sha256 = sha256_json(replacement.track)
    contracts = [
        contract
        for contract in (
            prepared.runtime_actions,
            *(replacement.runtime_actions for replacement in replacements),
        )
        if contract is not None
    ]
    receipt_dir = run_dir / "runtime-actions" / "receipts"
    receipts = (
        [
            _read_contract(path, RuntimeActionExecutionReceipt)
            for path in _evidence_paths(receipt_dir, "action-*.receipt.json")
        ]
        if receipt_dir.is_dir()
        else []
    )
    definitions: dict[str, tuple[Any, set[str]]] = {}
    for contract in contracts:
        contract_sha256 = sha256_json(contract)
        for step in contract.steps:
            existing = definitions.get(step.step_id)
            if existing is not None and existing[0] != step:
                raise PreparedMissionBindingError(f"RUNTIME_ACTION_STEP_ID_REUSED:{step.step_id}")
            if existing is None:
                definitions[step.step_id] = (step, {contract_sha256})
            else:
                # 修订保留同一动作时，修订之前已经完成的真实回执仍绑定原合同。
                existing[1].add(contract_sha256)
    superseded = {
        step_id
        for replacement in replacements
        for step_id in replacement.superseded_runtime_action_step_ids
    }
    expected = set(definitions) - superseded
    observed = {receipt.step_id: receipt for receipt in receipts}
    if len(observed) != len(receipts):
        raise PreparedMissionBindingError("RUNTIME_ACTION_RECEIPT_STEP_DUPLICATED")
    if not observed.keys() <= definitions.keys():
        raise PreparedMissionBindingError("RUNTIME_ACTION_RECEIPT_SET_MISMATCH")
    for step_id, receipt in observed.items():
        step, contract_hashes = definitions[step_id]
        binding_gates = {
            "time_order": (
                receipt.started_at.utcoffset() is not None
                and receipt.completed_at.utcoffset() is not None
                and receipt.completed_at >= receipt.started_at
            ),
            "execution_contract_hash": receipt.execution_contract_sha256 in contract_hashes,
            "step_hash": receipt.step_sha256 == sha256_json(step),
            "task_identity": receipt.task_id == step.task_id and receipt.action == step.action,
            "adapter_identity": (
                receipt.adapter_id == step.adapter_id
                and receipt.runtime_executor == step.runtime_executor
            ),
            "attempt_budget": receipt.attempts <= step.max_attempts,
            "success_evidence": (
                receipt.status != "accepted"
                or {item.casefold() for item in step.required_success_evidence}
                <= {item.casefold() for item in receipt.observed_success_evidence}
            ),
        }
        if not all(binding_gates.values()):
            failed = ",".join(name for name, accepted in binding_gates.items() if not accepted)
            raise PreparedMissionBindingError(f"RUNTIME_ACTION_RECEIPT_INVALID:{step_id}:{failed}")
    result = sorted(observed.values(), key=lambda item: (item.started_at, item.step_id)), expected
    return result


# 功能：
#   逐项绑定检查点定义、实际请求、模型决策与磁盘回执，不能用重复的成功响应凑够数量。
# 输入：
#   required：最终必须完成的检查点合同；contracts：本次运行实际采用的全部合同。
#   decisions：协调器返回的决策；run_dir：当前运行证据目录。
# 输出：
#   accepted：全部必需检查点真实通过且全部已发生决策均对应合法请求的联合结果。
def _checkpoint_receipts_passed(
    required: RuntimeCheckpointContract | None,
    contracts: list[RuntimeCheckpointContract | None],
    decisions: list[RuntimeCheckpointDecision],
    run_dir: Path,
) -> bool:
    if required is None or not required.checkpoints:
        return False
    required = _owned_contract(required)
    contracts = [_owned_contract(contract) for contract in contracts if contract is not None]
    decisions = [_owned_contract(decision) for decision in decisions]
    if any(contract.contract_id != required.contract_id for contract in contracts):
        return False
    definitions = {sha256_json(item) for contract in contracts for item in contract.checkpoints}
    required_definitions = {sha256_json(item) for item in required.checkpoints}
    if len({item.checkpoint_id for item in required.checkpoints}) != len(required.checkpoints):
        return False
    by_request = {decision.request_sha256: decision for decision in decisions}
    if len(by_request) != len(decisions):
        return False
    observed_requests: set[str] = set()
    accepted_definitions: set[str] = set()
    for path in _evidence_paths(run_dir / "checkpoints", "*.request.json"):
        request = _read_contract(path, RuntimeCheckpointRequest)
        digest = sha256_json(request)
        decision = by_request.get(digest)
        if decision is None:
            # 换计划时可留下尚未决策的旧请求；它不能贡献任何完成证据。
            continue
        checkpoint = request.checkpoint
        definition = sha256_json(checkpoint)
        decision_path = path.with_name(f"{checkpoint.checkpoint_id}.decision.json")
        if (
            digest in observed_requests
            or path.name != f"{checkpoint.checkpoint_id}.request.json"
            or request.contract_id != required.contract_id
            or definition not in definitions
            or not decision_path.is_file()
            or _read_contract(decision_path, RuntimeCheckpointDecision) != decision
            or not all_required_gates_passed(request.deterministic_gates)
            or decision.continue_authorized is not True
            or decision.assessment.action != "accept"
            or decision.model_call.role != "execution_monitor"
            or decision.model_call.output_sha256 != sha256_json(decision.assessment)
        ):
            return False
        observed_requests.add(digest)
        accepted_definitions.add(definition)
    accepted = observed_requests == by_request.keys() and (
        required_definitions <= accepted_definitions
    )
    return accepted


# 功能：
#   分块核对运行文件摘要，拒绝链接及读取期间替换，不把完整长日志载入内存。
# 输入：
#   path：资产、计划或日志文件。
# 输出：
#   digest：实际读取字节摘要。
def _file_sha256(path: Path) -> str:
    digest = hash_plugin_file(path, limit=8 * 1024**3)
    return digest


# 功能：
#   重新读取当前可执行任务包，核对用户确认、资产身份及路线/净空/轨迹相互绑定。
# 输入：
#   prepared_path：任务包；confirm_contract_id：用户确认 ID；semantic_path、vehicle_sdf：当前资产。
# 输出：
#   package：任务、三份制品路径及三份已验证合同组成的元组。
def _load_package(
    prepared_path: Path,
    confirm_contract_id: str,
    semantic_path: Path,
    vehicle_sdf: Path,
) -> tuple[PreparedMission, Path, Path, Path, GraphRoute, RouteClearanceReport, Px4Track]:
    prepared = _read_contract(prepared_path, PreparedMission)
    if prepared.schema_version != "dronedream.prepared-mission.v4":
        raise PreparedMissionBindingError("PREPARED_MISSION_SCHEMA_OBSOLETE")
    if confirm_contract_id != prepared.contract.contract_id:
        raise PreparedMissionBindingError("CONFIRMATION_CONTRACT_ID_MISMATCH")
    if prepared.status != "awaiting_confirmation":
        raise PreparedMissionBindingError("PREPARED_MISSION_NOT_CONFIRMABLE")
    if _file_sha256(semantic_path) != prepared.contract.map_semantic_sha256:
        raise PreparedMissionBindingError("SEMANTIC_ASSET_HASH_MISMATCH")
    if _file_sha256(vehicle_sdf) != prepared.contract.vehicle_sha256:
        raise PreparedMissionBindingError("VEHICLE_ASSET_HASH_MISMATCH")

    package_dir = prepared_path.parent
    route_path = package_dir / "08-execution-route.json"
    clearance_path = package_dir / "09-route-clearance.json"
    track_path = package_dir / "10-px4-track.json"
    route = _read_contract(route_path, GraphRoute)
    clearance = _read_contract(clearance_path, RouteClearanceReport)
    track = _read_contract(track_path, Px4Track)
    if route != prepared.execution_route or sha256_json(route) != clearance.route_sha256:
        raise PreparedMissionBindingError("ROUTE_PACKAGE_BINDING_MISMATCH")
    if clearance != prepared.route_clearance or not clearance.accepted:
        raise PreparedMissionBindingError("CLEARANCE_PACKAGE_BINDING_MISMATCH")
    if track != prepared.px4_track:
        raise PreparedMissionBindingError("TRACK_PACKAGE_BINDING_MISMATCH")
    package = prepared, route_path, clearance_path, track_path, route, clearance, track
    return package


# 功能：
#   验证任务旁车记录与当前确认包的会话、修订和摘要，随后导入生命周期账本。
# 输入：
#   prepared_path：任务位置；prepared：已验证任务；context_store：本地账本。
# 输出：
#   lifecycle：已导入的对应生命周期绑定。
def _execution_lifecycle_binding(
    *, prepared_path: Path, prepared: PreparedMission, context_store: ContextStore
) -> MissionLifecycleBinding:
    prepared_hash = _prepared_mission_sha256(prepared)
    sidecar_path = prepared_path.parent / "mission-lifecycle.json"
    if not sidecar_path.is_file():
        raise PreparedMissionBindingError("MISSION_LIFECYCLE_SIDECAR_REQUIRED")
    binding = _read_contract(sidecar_path, MissionLifecycleBinding)
    gates = {
        "conversation": (binding.thread.conversation_id == prepared.contract.conversation_id),
        "mission": binding.thread.mission_id == binding.plan_revision.mission_id,
        "current_revision": (
            binding.thread.current_plan_revision_id == binding.plan_revision.plan_revision_id
        ),
        "contract": binding.plan_revision.contract_id == prepared.contract.contract_id,
        "prepared_hash": (binding.plan_revision.prepared_mission_sha256 == prepared_hash),
    }
    if not all(gates.values()):
        raise PreparedMissionBindingError("MISSION_LIFECYCLE_SIDECAR_MISMATCH")
    lifecycle = context_store.lifecycle.import_binding(binding)
    return lifecycle


# 功能：
#   验证用户执行授权对应当前会话、修订、任务包及冻结插件集合。
# 输入：
#   authority_path：授权记录；prepared：任务包；lifecycle：生命周期绑定。
# 输出：
#   authority：通过绑定核对的授权对象，一次性消费仍由生命周期负责。
def _verified_execution_authority(
    *,
    authority_path: Path,
    prepared: PreparedMission,
    lifecycle: MissionLifecycleBinding,
) -> ModelHarnessExecutionAuthority:
    if not authority_path.is_file():
        raise PreparedMissionBindingError("EXECUTION_AUTHORITY_REQUIRED")
    try:
        authority = _read_contract(authority_path, ModelHarnessExecutionAuthority)
    except (OSError, ValueError) as error:
        raise PreparedMissionBindingError("EXECUTION_AUTHORITY_INVALID") from error
    gates = {
        "thread": authority.thread_id == prepared.contract.conversation_id,
        "revision": (authority.plan_revision_id == lifecycle.plan_revision.plan_revision_id),
        "contract": authority.contract_id == prepared.contract.contract_id,
        "prepared_hash": authority.prepared_mission_sha256 == _prepared_mission_sha256(prepared),
        "plugin_snapshot": (authority.plugin_snapshot_id == prepared.plugin_snapshot.snapshot_id),
        "plugin_catalog": (
            authority.plugin_catalog_sha256 == prepared.plugin_snapshot.catalog_sha256
        ),
    }
    if not all(gates.values()):
        failed = ",".join(name for name, passed in gates.items() if not passed)
        raise PreparedMissionBindingError(f"EXECUTION_AUTHORITY_MISMATCH:{failed}")
    return authority


# 功能：
#   将实际运行读取的文件摘要与确认包再次绑定，制品中途被修改不能只凭新摘要通过。
# 输入：
#   prepared、runtime：确认包与实际运行证据；route_path、clearance_path、track_path：本轮制品。
# 输出：
#   gates：文件、语义资产、机型及验证计划的联合绑定明细。
def _binding_gates(
    *,
    prepared: PreparedMission,
    runtime: Px4GazeboRunEvidence,
    route_path: Path,
    clearance_path: Path,
    track_path: Path,
) -> dict[str, bool]:
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
    gates = {
        "current_artifact_values_match_confirmed_package": (
            _read_contract(route_path, GraphRoute) == prepared.execution_route
            and _read_contract(clearance_path, RouteClearanceReport) == prepared.route_clearance
            and _read_contract(track_path, Px4Track) == prepared.px4_track
        ),
        "runtime_route_file_hash_matches_prepared_file": (
            runtime.artifacts.route_sha256 == _file_sha256(route_path)
        ),
        "runtime_track_file_hash_matches_prepared_file": (
            runtime.artifacts.track_sha256 == _file_sha256(track_path)
        ),
        "runtime_clearance_file_hash_matches_prepared_file": (
            runtime.artifacts.clearance_sha256 == _file_sha256(clearance_path)
        ),
        "runtime_semantic_file_hash_matches_contract": (
            runtime.artifacts.semantic_sha256 == prepared.contract.map_semantic_sha256
        ),
        "runtime_vehicle_file_hash_matches_contract": (
            runtime.artifacts.vehicle_sha256 == prepared.contract.vehicle_sha256
        ),
        "verification_plan_hash_bindings_match_prepared_mission": (verification_plan_bound),
    }
    return gates


# 功能：
#   逐一停止已经创建的协调器，部分启动失败也回收；清理失败不覆盖原始运行异常。
# 输入：
#   workers：本轮拥有的协调器；timeout_seconds：单协调器停止预算。
# 输出：
#   无：清空已尝试回收的集合，必要时抛出清理异常组。
def _stop_execution_workers(workers: list[Any], timeout_seconds: float) -> None:
    primary = sys.exc_info()[1]
    errors: list[BaseException] = []
    while workers:
        worker = workers.pop()
        try:
            worker.stop(timeout_seconds=timeout_seconds)
        except BaseException as error:
            errors.append(error)
    if errors:
        if primary is not None:
            primary.add_note(
                "Execution worker cleanup also failed: "
                + ", ".join(type(error).__name__ for error in errors)
            )
        else:
            raise BaseExceptionGroup("Execution worker cleanup failed", errors)


# 功能：
#   调用完成核验模型，先保存实际调用痕迹，再严格验收输出；无论成功失败都关闭自有端口。
# 输入：
#   provider、timeout_seconds：模型配置；instructions、artifact：本次核验输入。
#   context_id、conversation_id：调用与会话标识；chain、context_store：证据存储。
# 输出：
#   completion：独立的结构化核验结果及供应商回执。
def _invoke_completion_verifier(
    *,
    provider,
    timeout_seconds,
    instructions,
    artifact,
    context_id,
    conversation_id,
    chain,
    context_store,
):
    port = StructuredModelPort(provider, max_attempts=3, timeout_seconds=timeout_seconds)
    try:
        completion = copy.deepcopy(
            port.call(
                role="completion_verifier",
                output_type=CompletionAssessment,
                instructions=instructions,
                input_artifact=copy.deepcopy(artifact),
                context_id=context_id,
            )
        )
        receipt = {
            "record": completion.record.model_dump(mode="json"),
            "supporting_records": [
                record.model_dump(mode="json") for record in completion.supporting_records
            ],
        }
        chain.append("model.completion_verifier.received", receipt)
        context_store.append(
            conversation_id,
            role="tool",
            event_type="model.completion_verifier.received",
            payload=receipt,
        )
        if not isinstance(completion.artifact, CompletionAssessment):
            raise PreparedMissionBindingError("COMPLETION_OUTPUT_TYPE_INVALID")
        output = _owned_contract(completion.artifact)
        record = _owned_contract(completion.record)
        if (
            not isinstance(record, ModelCallRecord)
            or record.output_sha256 != sha256_json(output)
            or record.input_sha256 != sha256_json(artifact)
            or record.role != "completion_verifier"
            or record.output_schema != CompletionAssessment.__name__
        ):
            raise PreparedMissionBindingError("COMPLETION_MODEL_CALL_BINDING_MISMATCH")
        return completion
    except Exception as primary:
        # 没有供应商用量回执时不编造 token；仍记录调用/验收失败及原输入身份。
        try:
            chain.append("model.completion_verifier.failed", {
                "error_type": type(primary).__name__, "input_sha256": sha256_json(artifact),
            })
        except Exception as error:
            primary.add_note("Completion failure recording also failed: " + type(error).__name__)
        raise
    finally:
        primary = sys.exc_info()[1]
        try:
            port.close()
        except Exception as error:
            if primary is None:
                raise
            primary.add_note("Completion port cleanup also failed: " + type(error).__name__)


# 功能：
#   1. 绑定运行证据、检查点、动作回执与当前计划，模型不能覆盖固定失败门控。
#   2. 保存模型和插件实际回执，独占发布本次完成结果。
# 输入：
#   prepared、route、clearance、track 及其路径：确认的任务与几何制品。
#   runtime、offboard_timing：运行反馈。
#   run_dir、evidence_filename、result_filename：本次证据输出位置。
#   completion_provider、model_timeout_seconds、context_store：模型调用及会话存储。
#   checkpoint_decisions、expected_checkpoint_count：检查点响应及所需数量。
#   runtime_action_receipts、required_runtime_action_step_ids：实际动作及有效必做集合。
#   runtime_interruption_decisions：用户临时改令的实际决策。
# 输出：
#   result：本次运行或复验结果。
def _complete(
    *,
    prepared: PreparedMission,
    route_path: Path,
    clearance_path: Path,
    track_path: Path,
    route: GraphRoute,
    clearance: RouteClearanceReport,
    track: Px4Track,
    runtime: Px4GazeboRunEvidence,
    offboard_timing: dict[str, Any],
    run_dir: Path,
    completion_provider: ProviderName,
    context_store: ContextStore,
    model_timeout_seconds: float,
    evidence_filename: str,
    result_filename: str,
    checkpoint_decisions: list[RuntimeCheckpointDecision],
    runtime_action_receipts: list[RuntimeActionExecutionReceipt],
    required_runtime_action_step_ids: set[str],
    runtime_interruption_decisions: list[RuntimeInterruptionDecision],
    expected_checkpoint_count: int,
) -> SimulationWorkflowResult:
    prepared, route, clearance, track, runtime = map(
        _owned_contract, (prepared, route, clearance, track, runtime)
    )
    checkpoint_decisions = [_owned_contract(value) for value in checkpoint_decisions]
    runtime_action_receipts = [_owned_contract(value) for value in runtime_action_receipts]
    runtime_interruption_decisions = [
        _owned_contract(value) for value in runtime_interruption_decisions
    ]
    if (run_dir / result_filename).exists():
        raise FileExistsError("workflow result already exists")
    extensions = runtime_extension_registry(prepared)
    receipt_path = (run_dir / evidence_filename).parent / "plugin-hook-receipts.jsonl"
    plugin_receipts = list(prepared.plugin_hook_receipts)
    plugin_receipts.extend(
        receipt for decision in checkpoint_decisions for receipt in decision.plugin_hook_receipts
    )
    plugin_receipts.extend(
        receipt
        for decision in runtime_interruption_decisions
        for receipt in decision.plugin_hook_receipts
    )
    binding_gates = _binding_gates(
        prepared=prepared,
        runtime=runtime,
        route_path=route_path,
        clearance_path=clearance_path,
        track_path=track_path,
    )
    chain = EvidenceChain(run_dir / evidence_filename)
    chain.append(
        "workflow.prepared-mission",
        {
            "contract_id": prepared.contract.contract_id,
            "canonical_json_hash_domain": {
                "prepared_mission_sha256": _prepared_mission_sha256(prepared),
                "route_sha256": sha256_json(route),
                "track_sha256": sha256_json(track),
                "clearance_sha256": sha256_json(clearance),
            },
            "file_byte_hash_domain": {
                "route_sha256": _file_sha256(route_path),
                "track_sha256": _file_sha256(track_path),
                "clearance_sha256": _file_sha256(clearance_path),
            },
        },
    )
    chain.append("workflow.runtime", runtime.model_dump(mode="json"))
    for receipt in runtime_action_receipts:
        chain.append("runtime.domain-action", receipt.model_dump(mode="json"))
    for decision in checkpoint_decisions:
        chain.append(
            "model.execution_monitor",
            decision.model_dump(mode="json"),
        )
        context_store.append(
            prepared.contract.conversation_id,
            role="assistant",
            event_type="model.execution_monitor",
            payload=decision.model_dump(mode="json"),
        )
    for decision in runtime_interruption_decisions:
        chain.append("model.runtime_message_classifier", decision.model_dump(mode="json"))
        context_store.append(
            prepared.contract.conversation_id,
            role="assistant",
            event_type="model.runtime_message_classifier",
            payload=decision.model_dump(mode="json"),
        )
    context_store.append(
        prepared.contract.conversation_id,
        role="tool",
        event_type="simulation.runtime",
        payload=runtime.model_dump(mode="json"),
    )

    replacement_decisions = [
        decision
        for decision in runtime_interruption_decisions
        if decision.authorized_action == "hold_for_replan"
    ]
    adopted_replacements = _adopted_runtime_replacements(run_dir, prepared)
    adoption_count = len(adopted_replacements)
    checkpoint_contracts = [
        prepared.runtime_checkpoints,
        *(replacement.runtime_checkpoints for replacement in adopted_replacements),
    ]
    checkpoint_gate = _checkpoint_receipts_passed(
        checkpoint_contracts[-1], checkpoint_contracts, checkpoint_decisions, run_dir
    )
    if not adopted_replacements:
        checkpoint_gate = checkpoint_gate and len(checkpoint_decisions) == expected_checkpoint_count
    binding_gates["all_required_model_checkpoints_accepted"] = checkpoint_gate
    binding_gates["runtime_replacements_adopted_if_requested"] = adoption_count == len(
        replacement_decisions
    )
    binding_gates["hash_bound_user_confirmation_consumed"] = True
    observed_runtime_action_step_ids = {receipt.step_id for receipt in runtime_action_receipts}
    effective_runtime_actions_accepted = (
        required_runtime_action_step_ids <= observed_runtime_action_step_ids
        and all(
            receipt.status == "accepted"
            and not receipt.issue_codes
            and all_required_gates_passed(receipt.deterministic_gates)
            for receipt in runtime_action_receipts
        )
    )
    binding_gates["all_prepared_runtime_actions_accepted"] = effective_runtime_actions_accepted
    binding_gates["all_effective_runtime_actions_accepted"] = effective_runtime_actions_accepted
    try:
        runtime_evaluations, runtime_evaluation_receipts = extensions.invoke_multiple(
            "evaluation.runtime-gates",
            "evaluate_runtime",
            prepared=prepared.model_copy(deep=True),
            runtime=runtime.model_copy(deep=True),
            binding_gates=dict(binding_gates),
            checkpoint_decisions=copy.deepcopy(checkpoint_decisions),
            runtime_interruption_decisions=copy.deepcopy(runtime_interruption_decisions),
            run_dir=run_dir,
        )
        plugin_receipts.extend(runtime_evaluation_receipts)
        runtime_plugin_gates, normalized_runtime_evaluations = require_plugin_acceptance(
            runtime_evaluations,
            gate_prefix="plugin_runtime",
        )
        binding_gates.update(runtime_plugin_gates)
        completion_instructions, completion_prompt_receipts = augment_runtime_prompt(
            extensions,
            role="completion_verifier",
            instructions=COMPLETION_VERIFIER,
        )
        plugin_receipts.extend(completion_prompt_receipts)
    except ExtensionExecutionError as error:
        append_hook_receipts(receipt_path, [error.receipt])
        raise
    append_hook_receipts(
        receipt_path,
        [*runtime_evaluation_receipts, *completion_prompt_receipts],
    )
    deterministic_success = (
        runtime.status == "verified"
        and _active_runtime_gates_passed(runtime)
        and all(binding_gates.values())
    )
    completion = _invoke_completion_verifier(
        provider=completion_provider,
        timeout_seconds=model_timeout_seconds,
        chain=chain,
        context_store=context_store,
        conversation_id=prepared.contract.conversation_id,
        instructions=completion_instructions,
        artifact=_completion_verifier_input(
            prepared=prepared,
            route=route,
            clearance=clearance,
            track=track,
            runtime=runtime,
            offboard_timing=offboard_timing,
            binding_gates=binding_gates,
            normalized_runtime_evaluations=normalized_runtime_evaluations,
            checkpoint_decisions=checkpoint_decisions,
            runtime_action_receipts=runtime_action_receipts,
            runtime_interruption_decisions=runtime_interruption_decisions,
            deterministic_success=deterministic_success,
        ),
        context_id=f"{prepared.contract.conversation_id}::completion_verifier",
    )
    try:
        completion_output_receipts = validate_runtime_model_output(
            extensions,
            role="completion_verifier",
            expected_schema=CompletionAssessment.__name__,
            artifact=completion.artifact.model_copy(deep=True),
            record=completion.record.model_copy(deep=True),
        )
    except ExtensionExecutionError as error:
        append_hook_receipts(receipt_path, [error.receipt])
        raise
    plugin_receipts.extend(completion_output_receipts)
    append_hook_receipts(receipt_path, completion_output_receipts)
    completion_payload = {
        "artifact": completion.artifact.model_dump(mode="json"),
        "record": completion.record.model_dump(mode="json"),
    }
    context_store.append(
        prepared.contract.conversation_id,
        role="assistant",
        event_type="model.completion_verifier",
        payload=completion_payload,
    )
    chain.append("model.completion_verifier", completion_payload)
    try:
        exporter_outputs, exporter_receipts = extensions.invoke_multiple(
            "evidence.exporters",
            "export_evidence",
            run_dir=(run_dir / evidence_filename).parent,
            prepared=prepared.model_copy(deep=True),
            runtime=runtime.model_copy(deep=True),
            binding_gates=dict(binding_gates),
            plugin_evaluations=copy.deepcopy(normalized_runtime_evaluations),
            completion_assessment=completion.artifact.model_copy(deep=True),
        )
    except ExtensionExecutionError as error:
        append_hook_receipts(receipt_path, [error.receipt])
        raise
    plugin_receipts.extend(exporter_receipts)
    append_hook_receipts(receipt_path, exporter_receipts)
    for receipt in [
        *runtime_evaluation_receipts,
        *completion_prompt_receipts,
        *completion_output_receipts,
        *exporter_receipts,
    ]:
        chain.append("plugin.hook", receipt.model_dump(mode="json"))
    for output in exporter_outputs:
        chain.append("plugin.evidence-export", output)
    head = chain.read()[-1].record_sha256
    result = SimulationWorkflowResult(
        status=("verified" if deterministic_success and completion.artifact.accepted else "failed"),
        contract_id=prepared.contract.contract_id,
        prepared_mission_sha256=_prepared_mission_sha256(prepared),
        runtime_evidence=runtime,
        completion_assessment=completion.artifact,
        completion_model_call=completion.record,
        checkpoint_decisions=checkpoint_decisions,
        runtime_action_receipts=runtime_action_receipts,
        runtime_interruption_decisions=runtime_interruption_decisions,
        plugin_hook_receipts=plugin_receipts,
        workflow_evidence_chain_head=head,
    )
    publish_runtime_json(
        run_dir / result_filename,
        result,
        replace_existing=False,
        maximum_bytes=_EXECUTION_JSON_MAX_BYTES,
    )
    return result


# 功能：
#   1. 消费确认的任务修订并启动真实 Gazebo/PX4 执行，旁路处理检查点和临时改令。
#   2. 结束后核验实际结果，回收本轮拥有的工作进程。
# 输入：
#   prepared_path、execution_authority_path、confirm_contract_id：任务与用户授权。
#   run_dir：独立运行目录。
#   world_sdf、semantic_path、map_graph_path：本轮世界与语义、路径资产。
#   vehicle_sdf、vehicle_metadata_path：机型资产与物理描述。
#   controller_params_path、executor_path、px4_root、ros_workspace：控制器及仿真运行环境。
#   completion_provider、context_store、model_timeout_seconds：完成核验模型与状态存储。
#   checkpoint_provider、checkpoint_executor_path、checkpoint_timeout_seconds：检查点旁路。
#   runtime_interrupt_provider 及 runtime_*_seconds：临时改令与悬停时间预算。
#   local_navigation_*：本地/云端导航端口、周期、视觉和控制授权策略。
#   heading_policy、maximum_yaw_rate_deg_s：偏航控制限制；multimodal_*：训练观测记录配置。
#   local_policy_*_paths：模型包及资格凭据；development_payload_collection：仅仿真开发采集开关。
# 输出：
#   result：绑定真实运行证据的结果，不能仅凭模型说成功而通过。
def execute_prepared_mission(
    *,
    prepared_path: Path,
    execution_authority_path: Path,
    confirm_contract_id: str,
    run_dir: Path,
    world_sdf: Path,
    semantic_path: Path,
    vehicle_sdf: Path,
    controller_params_path: Path,
    executor_path: Path,
    px4_root: Path,
    ros_workspace: Path,
    completion_provider: ProviderName,
    context_store: ContextStore,
    model_timeout_seconds: float = 180.0,
    checkpoint_provider: ProviderName | None = None,
    checkpoint_executor_path: Path | None = None,
    checkpoint_timeout_seconds: float = 180.0,
    runtime_interrupt_provider: ProviderName | None = None,
    runtime_hold_timeout_seconds: float = 12.0,
    runtime_decision_timeout_seconds: float = 180.0,
    runtime_replan_hold_seconds: float = 60.0,
    local_navigation_provider: ProviderName | None = None,
    local_navigation_fallback_provider: ProviderName | None = None,
    local_navigation_model_timeout_seconds: float = 10.0,
    local_navigation_fallback_model_timeout_seconds: float = 10.0,
    local_navigation_period_seconds: float = 3.0,
    local_navigation_context_id: str | None = None,
    local_navigation_visual_enabled: bool = False,
    local_navigation_control_authority_required: bool = False,
    bounded_hybrid_control: bool = False,
    independent_route_control: bool = False,
    heading_policy: str = "measured-hold",
    maximum_yaw_rate_deg_s: float = 20.0,
    multimodal_dataset_root: Path | None = None,
    multimodal_flight_id: str | None = None,
    multimodal_dataset_maximum_mib: int = 5_120,
    multimodal_record_period_seconds: float = 0.1,
    local_policy_package_paths: tuple[Path, ...] = (),
    local_policy_qualification_paths: tuple[Path, ...] = (),
    local_policy_simulation_admission_paths: tuple[Path, ...] = (),
    local_policy_trial_path: Path | None = None,
    development_payload_collection: bool = False,
    map_graph_path: Path | None = None,
    vehicle_metadata_path: Path | None = None,
    simulation_map_fusion: bool = False,
) -> SimulationWorkflowResult:
    for name, value in {
        "model_timeout_seconds": model_timeout_seconds,
        "checkpoint_timeout_seconds": checkpoint_timeout_seconds,
        "runtime_hold_timeout_seconds": runtime_hold_timeout_seconds,
        "runtime_decision_timeout_seconds": runtime_decision_timeout_seconds,
        "runtime_replan_hold_seconds": runtime_replan_hold_seconds,
        "local_navigation_model_timeout_seconds": local_navigation_model_timeout_seconds,
        "local_navigation_fallback_model_timeout_seconds": (
            local_navigation_fallback_model_timeout_seconds
        ),
        "local_navigation_period_seconds": local_navigation_period_seconds,
    }.items():
        if type(value) not in (int, float) or not 0 < value <= 3600:
            raise ValueError(f"{name} must be finite and in (0, 3600]")
    if any(
        type(value) is not bool
        for value in (
            local_navigation_visual_enabled,
            local_navigation_control_authority_required,
            bounded_hybrid_control,
            development_payload_collection,
            simulation_map_fusion,
        )
    ):
        raise ValueError("execution mode flags must be boolean")
    if bounded_hybrid_control and (not local_navigation_control_authority_required
            or local_navigation_provider != "local-policy"):
        raise ValueError("HYBRID_REQUIRES_LOCAL_MODEL_CONTROL")
    from .route_control_mode import validate_route_control_mode
    validate_route_control_mode(enabled=independent_route_control,
        model_required=local_navigation_control_authority_required,
        provider=local_navigation_provider, hybrid=bounded_hybrid_control,
        training=development_payload_collection, fusion=simulation_map_fusion,
        heading=heading_policy)
    check_plain_plugin_path(run_dir)
    if run_dir.exists() and next(run_dir.iterdir(), None) is not None:
        raise FileExistsError("execution directory must be empty")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    (
        prepared,
        route_path,
        clearance_path,
        track_path,
        route,
        clearance,
        track,
    ) = _load_package(prepared_path, confirm_contract_id, semantic_path, vehicle_sdf)
    # 在创建执行生命周期和消费授权前检查所有运行钩子，避免进入执行后才发现插件无法重建。
    runtime_extension_registry(prepared)
    runtime_map_graph: MapAsset | None = None
    runtime_map_catalog = None
    runtime_vehicle: VehicleAsset | None = None
    if runtime_interrupt_provider is not None:
        if map_graph_path is None or vehicle_metadata_path is None:
            raise PreparedMissionBindingError("RUNTIME_REPLAN_ASSETS_REQUIRED")
        runtime_map_graph = _read_contract(map_graph_path, MapAsset)
        runtime_vehicle = _read_contract(vehicle_metadata_path, VehicleAsset)
        runtime_map_catalog = load_map_catalog(semantic_path, qualified_graph=runtime_map_graph)
        if sha256_json(runtime_map_graph) != prepared.contract.map_sha256:
            raise PreparedMissionBindingError("RUNTIME_MAP_GRAPH_HASH_MISMATCH")
        if runtime_vehicle.asset_id != prepared.contract.vehicle_asset_id:
            raise PreparedMissionBindingError("RUNTIME_VEHICLE_ID_MISMATCH")
    elif local_navigation_provider is not None:
        if vehicle_metadata_path is None:
            raise PreparedMissionBindingError("LOCAL_NAVIGATION_VEHICLE_METADATA_REQUIRED")
        runtime_vehicle = _read_contract(vehicle_metadata_path, VehicleAsset)
        if runtime_vehicle.asset_id != prepared.contract.vehicle_asset_id:
            raise PreparedMissionBindingError("RUNTIME_VEHICLE_ID_MISMATCH")
    if runtime_interrupt_provider is not None and (
        checkpoint_provider is None or checkpoint_executor_path is None
    ):
        raise ValueError("runtime interruption requires the checkpoint provider and executor")
    if local_navigation_fallback_provider == local_navigation_provider and (
        local_navigation_fallback_provider is not None
    ):
        raise ValueError("local navigation fallback provider must differ from primary")
    if local_navigation_control_authority_required and local_navigation_provider is None:
        raise ValueError("model control authority requires a local navigation provider")
    if local_navigation_provider is not None and vehicle_metadata_path is None:
        raise PreparedMissionBindingError("LOCAL_NAVIGATION_VEHICLE_METADATA_REQUIRED")
    if local_navigation_provider == "local-policy":
        if not local_policy_package_paths or not (
            local_policy_qualification_paths or local_policy_simulation_admission_paths
            or local_policy_trial_path
        ):
            raise PreparedMissionBindingError("LOCAL_POLICY_ARTIFACTS_REQUIRED")
    elif (
        local_policy_package_paths
        or local_policy_qualification_paths
        or local_policy_simulation_admission_paths
        or local_policy_trial_path
    ):
        raise PreparedMissionBindingError("LOCAL_POLICY_PROVIDER_REQUIRED")
    if local_policy_trial_path is not None:
        from .simulation_trial import validate_trial_configuration

        validate_trial_configuration(
            local_policy_trial_path, local_policy_package_paths, semantic_path,
            runtime_vehicle, qualifications=local_policy_qualification_paths,
            admissions=local_policy_simulation_admission_paths,
            fallback=local_navigation_fallback_provider,
            incompatible=development_payload_collection,
        )
    fusion_arguments = {}
    if simulation_map_fusion:
        if not independent_route_control and (
                local_policy_trial_path is None or not local_navigation_visual_enabled
                or not local_navigation_control_authority_required):
            raise ValueError("MAP_FUSION_REQUIRES_VISUAL_SIMULATION_TRIAL")
        from .simulation_fusion_runtime import prepare_simulation_fusion

        executor_path, fusion_arguments = prepare_simulation_fusion(
            run_dir=run_dir, world_sdf=world_sdf, semantic_path=semantic_path,
            executor_path=executor_path, px4_root=px4_root,
        )
    if development_payload_collection:
        if local_navigation_provider != "local-policy":
            raise PreparedMissionBindingError(
                "DEVELOPMENT_PAYLOAD_COLLECTION_LOCAL_POLICY_REQUIRED"
            )
        if not local_policy_simulation_admission_paths or local_policy_qualification_paths:
            raise PreparedMissionBindingError(
                "DEVELOPMENT_PAYLOAD_COLLECTION_SIMULATION_ADMISSION_ONLY"
            )
        if multimodal_dataset_root is None:
            raise PreparedMissionBindingError(
                "DEVELOPMENT_PAYLOAD_COLLECTION_MULTIMODAL_RECORDING_REQUIRED"
            )
    if (
        prepared.runtime_actions is not None
        and prepared.runtime_actions.steps
        and (checkpoint_provider is None or checkpoint_executor_path is None)
    ):
        raise ValueError("prepared runtime actions require the checkpoint provider and executor")

    lifecycle = _execution_lifecycle_binding(
        prepared_path=prepared_path, prepared=prepared, context_store=context_store
    )
    _verified_execution_authority(
        authority_path=execution_authority_path,
        prepared=prepared,
        lifecycle=lifecycle,
    )
    thread, plan_revision, execution_id = context_store.lifecycle.confirm_execution(
        conversation_id=prepared.contract.conversation_id,
        plan_revision_id=lifecycle.plan_revision.plan_revision_id,
        contract_id=prepared.contract.contract_id,
        prepared_mission_sha256=_prepared_mission_sha256(prepared),
    )
    checkpoint_decisions: list[RuntimeCheckpointDecision] = []
    runtime_action_receipts: list[RuntimeActionExecutionReceipt] = []
    required_runtime_action_step_ids: set[str] = set()
    runtime_interruption_decisions: list[RuntimeInterruptionDecision] = []
    expected_checkpoint_count = 0
    control_dir = run_dir / "runtime-control"
    runtime_session = None
    if runtime_interrupt_provider is not None:
        try:
            runtime_session = create_runtime_control_session(
                control_dir=control_dir,
                conversation_id=thread.conversation_id,
                mission_id=thread.mission_id,
                plan_revision_id=plan_revision.plan_revision_id,
                contract_id=prepared.contract.contract_id,
                execution_id=execution_id,
                prepared_mission_sha256=_prepared_mission_sha256(prepared),
            )
        except BaseException as primary:
            try:
                context_store.lifecycle.set_execution_state(
                    conversation_id=prepared.contract.conversation_id,
                    execution_id=execution_id,
                    state="failed",
                )
            except Exception as error:
                primary.add_note("Lifecycle failure recording also failed: " + type(error).__name__)
            raise

    result: SimulationWorkflowResult | None = None
    owned_workers: list[Any] = []
    try:
        if checkpoint_provider is not None:
            if checkpoint_executor_path is None:
                raise ValueError("checkpoint_executor_path is required with checkpoint_provider")
            checkpoint_contract = checkpoint_contract_for(prepared)
            expected_checkpoint_count = len(checkpoint_contract.checkpoints)
            with tempfile.TemporaryDirectory(
                prefix="dronedream-checkpoints-", dir=run_dir.parent
            ) as temporary:
                checkpoint_path = Path(temporary) / "runtime-checkpoints.json"
                checkpoint_path.write_text(
                    checkpoint_contract.model_dump_json(indent=2) + "\n",
                    encoding="utf-8",
                )
                runtime_actions_path = Path(temporary) / "runtime-actions.json"
                if prepared.runtime_actions is not None:
                    runtime_actions_path.write_text(
                        prepared.runtime_actions.model_dump_json(indent=2) + "\n",
                        encoding="utf-8",
                    )
                checkpoint_coordinator = CheckpointCoordinator(
                    prepared=prepared,
                    run_dir=run_dir,
                    provider=checkpoint_provider,
                    abort_file=run_dir / "live_abort.request.json",
                    model_timeout_seconds=model_timeout_seconds,
                )
                owned_workers.append(checkpoint_coordinator)
                interruption_coordinator = (
                    RuntimeInterruptionCoordinator(
                        prepared=prepared,
                        session=runtime_session,
                        control_dir=control_dir,
                        provider=runtime_interrupt_provider,
                        abort_file=run_dir / "live_abort.request.json",
                        lifecycle_db_path=context_store.path,
                        model_timeout_seconds=model_timeout_seconds,
                        map_graph=runtime_map_graph,
                        map_catalog=runtime_map_catalog,
                        semantic_path=semantic_path,
                        vehicle=runtime_vehicle,
                    )
                    if runtime_session is not None and runtime_interrupt_provider is not None
                    else None
                )
                if interruption_coordinator is not None:
                    owned_workers.append(interruption_coordinator)
                checkpoint_coordinator.start()
                if interruption_coordinator is not None:
                    interruption_coordinator.start()
                extra_args = [
                    "--base-executor",
                    str(executor_path),
                ]
                try:
                    raw_runtime = run_px4_gazebo_track(
                        run_dir=run_dir,
                        world_sdf=world_sdf,
                        semantic_path=semantic_path,
                        vehicle_sdf=vehicle_sdf,
                        route_path=route_path,
                        track_path=track_path,
                        clearance_path=clearance_path,
                        controller_params_path=controller_params_path,
                        px4_root=px4_root,
                        executor_path=checkpoint_executor_path,
                        ros_workspace=ros_workspace,
                        contract_id=prepared.contract.contract_id,
                        executor_extra_args=extra_args,
                        checkpoint_contract_path=checkpoint_path,
                        runtime_control_dir=control_dir if runtime_session is not None else None,
                        checkpoint_timeout_seconds=checkpoint_timeout_seconds,
                        runtime_hold_timeout_seconds=runtime_hold_timeout_seconds,
                        runtime_decision_timeout_seconds=runtime_decision_timeout_seconds,
                        runtime_replan_hold_seconds=runtime_replan_hold_seconds,
                        runtime_action_contract_path=(
                            runtime_actions_path
                            if prepared.runtime_actions is not None
                            and prepared.runtime_actions.steps
                            else None
                        ),
                        vehicle_metadata_path=vehicle_metadata_path,
                        local_navigation_provider=local_navigation_provider,
                        local_navigation_fallback_provider=(local_navigation_fallback_provider),
                        local_navigation_model_timeout_seconds=(
                            local_navigation_model_timeout_seconds
                        ),
                        local_navigation_fallback_model_timeout_seconds=(
                            local_navigation_fallback_model_timeout_seconds
                        ),
                        local_navigation_period_seconds=local_navigation_period_seconds,
                        local_navigation_context_id=local_navigation_context_id,
                        local_navigation_visual_enabled=local_navigation_visual_enabled,
                        local_navigation_control_authority_required=(
                            local_navigation_control_authority_required
                        ),
                        bounded_hybrid_control=bounded_hybrid_control,
                        independent_route_control=independent_route_control,
                        heading_policy=heading_policy,
                        maximum_yaw_rate_deg_s=maximum_yaw_rate_deg_s,
                        multimodal_dataset_root=multimodal_dataset_root,
                        multimodal_flight_id=multimodal_flight_id,
                        multimodal_dataset_maximum_mib=(multimodal_dataset_maximum_mib),
                        multimodal_record_period_seconds=(multimodal_record_period_seconds),
                        local_policy_package_paths=local_policy_package_paths,
                        local_policy_qualification_paths=(local_policy_qualification_paths),
                        local_policy_simulation_admission_paths=(
                            local_policy_simulation_admission_paths
                        ),
                        local_policy_trial_path=local_policy_trial_path,
                        development_payload_collection=development_payload_collection,
                        # Payload-corpus flights need the onboard RGB/depth
                        # sensors recorded below, but not the separate 1280x720
                        # spectator camera.  Disabling only that unconsumed
                        # render stream keeps the simulator close to real time
                        # without weakening any flight or dataset gate.
                        live_camera_enabled=not development_payload_collection,
                        **fusion_arguments,
                    )
                finally:
                    _stop_execution_workers(owned_workers, model_timeout_seconds + 10.0)
                if checkpoint_coordinator.error is not None:
                    raise RuntimeError("checkpoint coordinator failed") from (
                        checkpoint_coordinator.error
                    )
                checkpoint_decisions = list(checkpoint_coordinator.decisions)
                if interruption_coordinator is not None:
                    if interruption_coordinator.error is not None:
                        raise RuntimeError("runtime interruption coordinator failed") from (
                            interruption_coordinator.error
                        )
                    runtime_interruption_decisions = list(interruption_coordinator.decisions)
        else:
            raw_runtime = run_px4_gazebo_track(
                run_dir=run_dir,
                world_sdf=world_sdf,
                semantic_path=semantic_path,
                vehicle_sdf=vehicle_sdf,
                route_path=route_path,
                track_path=track_path,
                clearance_path=clearance_path,
                controller_params_path=controller_params_path,
                px4_root=px4_root,
                executor_path=executor_path,
                ros_workspace=ros_workspace,
                contract_id=prepared.contract.contract_id,
                vehicle_metadata_path=vehicle_metadata_path,
                local_navigation_provider=local_navigation_provider,
                local_navigation_fallback_provider=local_navigation_fallback_provider,
                local_navigation_model_timeout_seconds=(local_navigation_model_timeout_seconds),
                local_navigation_fallback_model_timeout_seconds=(
                    local_navigation_fallback_model_timeout_seconds
                ),
                local_navigation_period_seconds=local_navigation_period_seconds,
                local_navigation_context_id=local_navigation_context_id,
                local_navigation_visual_enabled=local_navigation_visual_enabled,
                local_navigation_control_authority_required=(
                    local_navigation_control_authority_required
                ),
                bounded_hybrid_control=bounded_hybrid_control,
                independent_route_control=independent_route_control,
                heading_policy=heading_policy,
                maximum_yaw_rate_deg_s=maximum_yaw_rate_deg_s,
                multimodal_dataset_root=multimodal_dataset_root,
                multimodal_flight_id=multimodal_flight_id,
                multimodal_dataset_maximum_mib=multimodal_dataset_maximum_mib,
                multimodal_record_period_seconds=multimodal_record_period_seconds,
                local_policy_package_paths=local_policy_package_paths,
                local_policy_qualification_paths=local_policy_qualification_paths,
                local_policy_simulation_admission_paths=(local_policy_simulation_admission_paths),
                local_policy_trial_path=local_policy_trial_path,
                development_payload_collection=development_payload_collection,
                live_camera_enabled=not development_payload_collection,
                **fusion_arguments,
            )
        runtime = Px4GazeboRunEvidence.model_validate_json(
            encode_json(raw_runtime, limit=_EXECUTION_JSON_MAX_BYTES, node_limit=1_000_000),
            strict=True,
        )
        offboard_timing = read_runtime_object(
            run_dir / "offboard_timing.json", maximum_bytes=16 * 1024 * 1024
        )
        (
            runtime_action_receipts,
            required_runtime_action_step_ids,
        ) = _load_runtime_action_receipts(prepared, run_dir)
        result = _complete(
            prepared=prepared,
            route_path=route_path,
            clearance_path=clearance_path,
            track_path=track_path,
            route=route,
            clearance=clearance,
            track=track,
            runtime=runtime,
            offboard_timing=offboard_timing,
            run_dir=run_dir,
            completion_provider=completion_provider,
            context_store=context_store,
            model_timeout_seconds=model_timeout_seconds,
            evidence_filename="workflow-evidence.jsonl",
            result_filename="workflow-result.json",
            checkpoint_decisions=checkpoint_decisions,
            runtime_action_receipts=runtime_action_receipts,
            required_runtime_action_step_ids=required_runtime_action_step_ids,
            runtime_interruption_decisions=runtime_interruption_decisions,
            expected_checkpoint_count=expected_checkpoint_count,
        )
        context_store.lifecycle.set_execution_state(
            conversation_id=prepared.contract.conversation_id,
            execution_id=execution_id,
            state="completed" if result.status == "verified" else "failed",
        )
        return result
    except BaseException as primary:
        try:
            current = context_store.lifecycle.get_thread(prepared.contract.conversation_id)
            if current is not None and current.active_execution_id == execution_id:
                context_store.lifecycle.set_execution_state(
                    conversation_id=prepared.contract.conversation_id,
                    execution_id=execution_id,
                    state="failed",
                )
        except Exception as error:
            primary.add_note("Lifecycle failure recording also failed: " + type(error).__name__)
        raise
    finally:
        try:
            _stop_execution_workers(owned_workers, model_timeout_seconds + 10.0)
        finally:
            if runtime_session is not None:
                primary = sys.exc_info()[1]
                try:
                    close_runtime_control_session(control_dir)
                except Exception as error:
                    if primary is None:
                        raise
                    primary.add_note("Control session cleanup also failed: " + type(error).__name__)


# 功能：
#   只对已经结束且与原结果一致的运行再做云端核验，不重飞、不改写原始门控或原结果。
# 输入：
#   prepared_path、confirm_contract_id：原确认任务；run_dir：原运行证据。
#   semantic_path、vehicle_sdf：原资产；completion_provider、model_timeout_seconds：核验模型预算。
#   context_store：会话记录存储。
# 输出：
#   result：独立 reviews 子目录中的本次核验结果。
def reverify_prepared_run(
    *,
    prepared_path: Path,
    confirm_contract_id: str,
    run_dir: Path,
    semantic_path: Path,
    vehicle_sdf: Path,
    completion_provider: ProviderName,
    context_store: ContextStore,
    model_timeout_seconds: float = 180.0,
) -> SimulationWorkflowResult:
    (
        prepared,
        route_path,
        clearance_path,
        track_path,
        route,
        clearance,
        track,
    ) = _load_package(prepared_path, confirm_contract_id, semantic_path, vehicle_sdf)
    runtime = _read_contract(run_dir / "mission_evidence.json", Px4GazeboRunEvidence)
    offboard_timing = read_runtime_object(
        run_dir / "offboard_timing.json", maximum_bytes=16 * 1024 * 1024
    )
    prior_result_path = run_dir / "workflow-result.json"
    if not prior_result_path.is_file():
        raise PreparedMissionBindingError("ORIGINAL_WORKFLOW_RESULT_MISSING")
    prior_result = _read_contract(prior_result_path, SimulationWorkflowResult)
    if prior_result.contract_id != prepared.contract.contract_id:
        raise PreparedMissionBindingError("ORIGINAL_WORKFLOW_CONTRACT_MISMATCH")
    if prior_result.prepared_mission_sha256 != _prepared_mission_sha256(prepared):
        raise PreparedMissionBindingError("ORIGINAL_WORKFLOW_PREPARED_HASH_MISMATCH")
    if runtime != prior_result.runtime_evidence:
        raise PreparedMissionBindingError("ORIGINAL_RUNTIME_EVIDENCE_MISMATCH")
    original_chain = EvidenceChain(run_dir / "workflow-evidence.jsonl").read()
    if (
        not original_chain
        or original_chain[-1].record_sha256 != prior_result.workflow_evidence_chain_head
    ):
        raise PreparedMissionBindingError("ORIGINAL_WORKFLOW_EVIDENCE_CHAIN_MISMATCH")
    for item in [*runtime.artifacts.px4_ulogs, *runtime.artifacts.model_navigation_visual_frames]:
        source_path = run_dir / item.path
        check_plain_plugin_path(source_path)
        ulog_path = source_path.resolve()
        try:
            ulog_path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise PreparedMissionBindingError("PX4_ULOG_PATH_ESCAPES_RUN") from exc
        if (
            not ulog_path.is_file()
            or ulog_path.stat().st_size != item.size_bytes
            or _file_sha256(ulog_path) != item.sha256
        ):
            raise PreparedMissionBindingError("PX4_ULOG_BINDING_MISMATCH")
    receipts, required_runtime_action_step_ids = _load_runtime_action_receipts(prepared, run_dir)
    if receipts != prior_result.runtime_action_receipts:
        raise PreparedMissionBindingError("ORIGINAL_ACTION_RECEIPTS_MISMATCH")
    review_path = Path("reviews") / ("review-" + uuid4().hex)
    result = _complete(
        prepared=prepared,
        route_path=route_path,
        clearance_path=clearance_path,
        track_path=track_path,
        route=route,
        clearance=clearance,
        track=track,
        runtime=runtime,
        offboard_timing=offboard_timing,
        run_dir=run_dir,
        completion_provider=completion_provider,
        context_store=context_store,
        model_timeout_seconds=model_timeout_seconds,
        evidence_filename=(review_path / "workflow-evidence.jsonl").as_posix(),
        result_filename=(review_path / "workflow-result.json").as_posix(),
        checkpoint_decisions=prior_result.checkpoint_decisions,
        runtime_action_receipts=prior_result.runtime_action_receipts,
        required_runtime_action_step_ids=required_runtime_action_step_ids,
        runtime_interruption_decisions=prior_result.runtime_interruption_decisions,
        expected_checkpoint_count=len(checkpoint_contract_for(prepared).checkpoints),
    )
    return result
