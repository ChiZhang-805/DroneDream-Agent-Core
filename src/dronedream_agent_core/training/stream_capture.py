"""Bounded streaming imitation evidence, deliberately not a PPO transition.

The policy can issue a new proposal before the preceding one has propagated
through sensing and actuation. Therefore adjacent inputs are not claimed to be
the result of one action. Only grounded DAgger supervision consumes these
records; no reward, next-state dynamics target or flight qualification is made.
"""

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pydantic import TypeAdapter

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from ..contracts import (
    NormalizedPilotControl,
    RuntimeLocalSafetyCommand,
    RuntimeLocalSafetyObservation,
)
from ..control_execution_evidence import ControlApplicationRecord
from ..hashing import sha256_json
from ..pilot_control_mapping import physical_pilot_request
from ..plugin_files import read_plugin_file
from ..runtime_evidence import (
    control_evidence_inventory,
    navigation_evidence_inventory,
    runtime_evidence_inventory_complete,
)
from .capture_archive import (
    MAXIMUM_ARCHIVE_BYTES,
    MAXIMUM_CAPTURE_BYTES,
    MAXIMUM_CAPTURE_JSON_NODES,
)
from .executed_control import executed_training_action
from .flight_environment import FlightObservation, PilotAction, SafetyPositionControl
from .observations import PreparedTrainingInput
from .policy_exchange import TrainingProposal
from .runtime_evidence import read_object


@dataclass(frozen=True)
class StreamActionCapture:
    """Original actor input/proposal plus a separate label-only simulator witness."""
    observation: FlightObservation
    request: dict
    proposal: TrainingProposal
    # Privileged simulation witness is label-only, never passed to the actor.
    initial_witness: RuntimeLocalSafetyObservation


_CAPTURE = TypeAdapter(StreamActionCapture)


@dataclass(frozen=True)
class PackedStreamCapture:
    """Owned immutable bytes retained during flight; no mutable observation alias."""
    content: bytes
    sha256: str

    # 功能：
    #   1. 核对捕获摘要并严格解码，重新编译原输入，检查提案与专家身份。
    #   2. 独立见证必须是健康且最多早一百毫秒的仿真来源，不能进入学生输入。
    # 输入：
    #   self：保存不可变内容与摘要的连续流捕获。
    # 输出：
    #   capture：来源绑定验证后的捕获，不代表动作已被执行。
    def unpack(self):
        if type(self.content) is not bytes or not 0 < len(self.content) <= MAXIMUM_CAPTURE_BYTES:
            raise ValueError("STREAM_CAPTURE_SIZE_INVALID")
        if hashlib.sha256(self.content).hexdigest() != self.sha256:
            raise ValueError("STREAM_CAPTURE_CONTENT_CHANGED")
        decode_json(self.content, limit=MAXIMUM_CAPTURE_BYTES,
                    node_limit=MAXIMUM_CAPTURE_JSON_NODES)
        capture = _CAPTURE.validate_json(self.content)
        prepared = PreparedTrainingInput.from_request(capture.request)
        exclude = {"visual_features", "source_visual_sha256"}
        if (
            prepared.sample.model_dump(exclude=exclude)
            != capture.observation.sample.model_dump(exclude=exclude)
            or capture.proposal.request_sha256 != prepared.request_sha256
            or capture.proposal.expert_role != capture.observation.sample.navigation_expert_role
        ):
            raise ValueError("STREAM_CAPTURE_INPUT_PROPOSAL_MISMATCH")
        witness = capture.initial_witness
        source_ms = capture.observation.sample.temporal_evidence.observed_at_unix_ms
        if (
            witness.source != "simulation-ground-truth"
            or not witness.stream_healthy
            or witness.stream_age_seconds > 0.1
            or not 0 <= source_ms - witness.observed_at_unix_ms <= 100
        ):
            raise ValueError("STREAM_CAPTURE_INITIAL_WITNESS_INVALID")
        return capture


# 功能：
#   将已验证的连续流提案保存为有界不可变字节，不推断动作接纳或奖励。
# 输入：
#   capture：包含原观测、请求、提案与独立见证的捕获对象。
# 输出：
#   packed：规范类型序列化内容及其摘要。
def pack_stream_capture(capture: StreamActionCapture) -> PackedStreamCapture:
    if type(capture) is not StreamActionCapture:
        raise ValueError("STREAM_CAPTURE_TYPE_INVALID")
    content = _CAPTURE.dump_json(capture, warnings="error")
    if not 0 < len(content) <= MAXIMUM_CAPTURE_BYTES:
        raise ValueError("STREAM_CAPTURE_SIZE_INVALID")
    packed = PackedStreamCapture(content, hashlib.sha256(content).hexdigest())
    return packed


# 功能：
#   从归档动作记录重建捕获，核对独立存储的捕获摘要后再检查来源。
# 输入：
#   row：带 capture 与 capture_sha256 的归档对象。
# 输出：
#   result：不可变捕获包及其已验证解码对象的二元组。
def stream_capture_from_record(row):
    if type(row) is not dict or type(row.get("capture")) is not dict:
        raise ValueError("STREAM_CAPTURE_RECORD_NOT_OBJECT")
    content = encode_json(row["capture"], limit=MAXIMUM_CAPTURE_BYTES,
                          node_limit=MAXIMUM_CAPTURE_JSON_NODES)
    packed = pack_stream_capture(_CAPTURE.validate_json(content))
    if packed.sha256 != row["capture_sha256"]:
        raise ValueError("STREAM_ANNOTATION_CAPTURE_CONTENT_CHANGED")
    result = packed, packed.unpack()
    return result


# 功能：
#   验证独立存储的提案载荷，不制造执行回执或实际控制证明。
# 输入：
#   payload：已经解析成 JSON 数据的提案捕获对象。
# 输出：
#   capture：通过类型与来源绑定检查的捕获。
def stream_capture_from_payload(payload):
    content = encode_json(payload, limit=MAXIMUM_CAPTURE_BYTES,
                          node_limit=MAXIMUM_CAPTURE_JSON_NODES)
    capture = pack_stream_capture(_CAPTURE.validate_json(content)).unpack()
    return capture


# 功能：
#   有界读取同一普通文件的完整归档对象，拒绝读取期替换、链接和含歧义的 JSON。
# 输入：
#   path：离线流记录路径。
# 输出：
#   row：通过文件身份、预算与严格 JSON 检查的对象。
def read_stream_record(path):
    # 归档包含完整历史，使用独立的八 MiB 预算，不放宽实时控制消息上限。
    content = read_plugin_file(path, limit=2 * MAXIMUM_CAPTURE_BYTES)
    row = decode_json(content, limit=2 * MAXIMUM_CAPTURE_BYTES,
                      node_limit=MAXIMUM_CAPTURE_JSON_NODES)
    if type(row) is not dict:
        raise ValueError("STREAM_CAPTURE_RECORD_NOT_OBJECT")
    return row


@dataclass(frozen=True)
class StreamControlVisit:
    """Actual action attribution for imitation, with no assumed causal next-state reward."""
    observation: FlightObservation
    proposal: PilotAction
    applied_action: PilotAction | SafetyPositionControl
    command_sha256: str
    application_sha256: str
    capture_sha256: str
    safety_intervened: bool


# 功能：
#   检查原生落地回执与仅仿真重置标记，禁止未停止或非仿真数据进入标注。
# 输入：
#   episode：需要验证的回合目录。
# 输出：
#   result：已核对的重置记录和终止生命周期记录。
def require_grounded_stream(episode: Path):
    terminal = read_object(episode / "flight/simulation/native-terminal-lifecycle.json")
    if (
        terminal.get("terminal_state") != "ON_GROUND"
        or terminal.get("landing_confirmed") is not True
        or terminal.get("safe_to_stop_watchdog") is not True
    ):
        raise ValueError("STREAM_CAPTURE_NATIVE_LANDING_NOT_CONFIRMED")
    reset = read_object(episode / "reset.json")
    if reset.get("simulation_only") is not True or reset.get("qualification_granted") is not False:
        raise ValueError("STREAM_CAPTURE_NOT_SIMULATION_COLLECTION")
    result = reset, terminal
    return result


# 功能：
#   离线逐行解析有界且身份固定的账本，拒绝截断尾行、过量记录和有歧义的 JSON。
# 输入：
#   path：完整执行账本路径。
# 输出：
#   row：依原顺序逐次产生的严格 JSON 对象。
def _records(path: Path):
    # 先持有同一次读取的字节；不能把每行有界误当作整份文件也有界。
    content = read_plugin_file(path, limit=MAXIMUM_ARCHIVE_BYTES)
    with BytesIO(content) as stream:
        count = 0
        while line := stream.readline(MAXIMUM_CAPTURE_BYTES + 1):
            count += 1
            if count > 100_000 or len(line) > MAXIMUM_CAPTURE_BYTES or not line.endswith(b"\n"):
                raise ValueError("STREAM_CAPTURE_LEDGER_INCOMPLETE_OR_OVERSIZED")
            row = decode_json(line, limit=MAXIMUM_CAPTURE_BYTES,
                              node_limit=MAXIMUM_CAPTURE_JSON_NODES)
            if type(row) is not dict:
                raise ValueError("STREAM_CAPTURE_LEDGER_NOT_OBJECT")
            yield row


# 功能：
#   1. 确认两类运行时写入器已排空，再把本批提案与完整实际执行账本关联。
#   2. 同一决策只取第一次真实接纳，不用后续刷新获得更有利的延迟。
# 输入：
#   simulation：本回合仿真证据目录。
#   call_ids：最多二百五十六个待匹配模型调用标识。
# 输出：
#   joined：调用标识到原始命令和首次执行回执的映射。
def grounded_control_index(simulation: Path, call_ids: set[str]):
    if (type(call_ids) not in (set, frozenset) or not 1 <= len(call_ids) <= 256
            or any(type(value) is not str or not value for value in call_ids)):
        raise ValueError("STREAM_CAPTURE_CALL_SET_INVALID")
    call_ids = frozenset(call_ids)
    for counts, summary_path in (
        (
            control_evidence_inventory(simulation),
            simulation / "runtime-state/control-application-writer.json",
        ),
        (
            navigation_evidence_inventory(simulation),
            simulation / "runtime-evidence-writer-summary.json",
        ),
    ):
        if not runtime_evidence_inventory_complete(
            read_object(summary_path), artifact_counts=counts, record_count=sum(counts.values())
        ):
            raise ValueError("STREAM_CAPTURE_RUNTIME_EVIDENCE_NOT_DRAINED")
    commands = {}
    for row in _records(simulation / "depth-local-safety-history.jsonl"):
        raw = row.get("command")
        if isinstance(raw, dict) and raw.get("model_call_id") in call_ids:
            command = RuntimeLocalSafetyCommand.model_validate(raw)
            commands[sha256_json(command)] = command
            if len(commands) > 4096:
                raise ValueError("STREAM_CAPTURE_COMMAND_BINDINGS_EXCEED_BOUND")
    joined, sequence, last_time = {}, 0, -1
    for raw in _records(simulation / "runtime-state/control-applications.jsonl"):
        application = ControlApplicationRecord.model_validate(raw)
        if application.sequence != sequence + 1 or application.accepted_at_unix_ms < last_time:
            raise ValueError("STREAM_CAPTURE_EXECUTION_ORDER_INVALID")
        sequence, last_time = application.sequence, application.accepted_at_unix_ms
        command = commands.get(application.command_sha256)
        if command is not None:
            joined.setdefault(command.model_call_id, (command, application))
    return joined


# 功能：
#   把同一捕获与实际传输关联，复核四轴物理缩放，并区分模型动作和安全替换。
# 输入：
#   capture：准备关联的解码捕获。
#   packed：该捕获对应的不可变内容与摘要。
#   command：运行时发布的安全控制命令。
#   application：执行器实际接纳的传输回执。
# 输出：
#   visit：具有来源摘要、真实动作和安全干预标记的控制访问记录。
def stream_visit(capture, packed, command, application) -> StreamControlVisit:
    verified = packed.unpack()
    if capture != verified:
        raise ValueError("STREAM_CAPTURE_DECODED_CONTENT_CHANGED")
    capture = verified
    snapshot = capture.request["snapshot"]
    if command.model_call_id != "model-" + sha256_json(capture.proposal)[:24] or (
        command.model_navigation_snapshot_sha256 is not None
        and command.model_navigation_snapshot_sha256 != snapshot["snapshot_sha256"]
    ):
        raise ValueError("STREAM_CAPTURE_ACTUAL_CONTROL_SOURCE_MISMATCH")
    proposed = capture.proposal.action
    intent = command.requested_control_intent
    if intent is not None:
        if (
            proposed.mode != "pilot-control"
            or intent.control_origin != "continuous-model-output"
            or intent.yaw_control_mode != "model-rate"
            or intent.source_expert != capture.proposal.expert_role
        ):
            raise ValueError("STREAM_CAPTURE_PROPOSAL_CONTROL_MISMATCH")
        expected = physical_pilot_request(
            NormalizedPilotControl(
                **dict(
                    zip(
                        ("forward_axis", "right_axis", "up_axis", "yaw_axis"),
                        proposed.axes,
                        strict=True,
                    )
                )
            ),
            capture.observation.sample.pilot_control_limits,
            harness_scale=intent.harness_control_scale,
        )
        actual = (
            intent.forward_velocity_mps,
            intent.right_velocity_mps,
            intent.up_velocity_mps,
            intent.yaw_rate_dps,
        )
        if any(abs(a - b) > 1e-6 for a, b in zip(actual, expected, strict=True)):
            raise ValueError("STREAM_CAPTURE_PROPOSAL_PHYSICAL_REQUEST_MISMATCH")
    elif application.transport == "velocity-ned":
        raise ValueError("STREAM_CAPTURE_MOTION_INTENT_MISSING")
    applied, expired = executed_training_action(
        snapshot, command, application, limits=capture.observation.sample.pilot_control_limits
    )
    if application.transport == "velocity-ned" and not application.model_authorized:
        raise ValueError("STREAM_CAPTURE_MOTION_NOT_MODEL_AUTHORIZED")
    changed = (
        expired
        or isinstance(applied, SafetyPositionControl)
        or applied.mode != proposed.mode
        or any(abs(a - b) > 1e-6 for a, b in zip(applied.axes, proposed.axes, strict=True))
    )
    visit = StreamControlVisit(
        capture.observation,
        proposed,
        applied,
        sha256_json(command),
        sha256_json(application),
        packed.sha256,
        changed,
    )
    return visit


# 功能：
#   1. 确认落地后保存原始提案，再依据实际账本生成已执行访问和缺失提案清单。
#   2. 发布采用调用方的独占写入器；中途失败可能保留部分提案，不伪装成原子事务。
# 输入：
#   episode：本次回合目录。
#   captures：最多二百五十六份有界不可变捕获。
#   write_new：只创建新文件、不会覆盖既有证据的发布回调。
# 输出：
#   visits：具有实际接纳依据的访问列表，不包含未执行提案。
def finalize_stream_captures(episode: Path, captures, *, write_new):
    reset, terminal = require_grounded_stream(episode)
    if (
        type(captures) not in (list, tuple) or not 1 <= len(captures) <= 256
        or any(type(c) is not PackedStreamCapture or type(c.content) is not bytes for c in captures)
    ):
        raise ValueError("STREAM_CAPTURE_ARCHIVE_INVALID")
    captures = tuple(captures)
    if sum(len(c.content) for c in captures) > MAXIMUM_ARCHIVE_BYTES:
        raise ValueError("STREAM_CAPTURE_ARCHIVE_INVALID")
    ids, source_files, identities = [], {}, set()
    # 实际账本不完整时仍可保留原始提案，但不能把提案升级成已执行证据。
    for index, packed in enumerate(captures):
        capture = packed.unpack()
        observation = capture.observation
        if (
            observation.episode_id != episode.name
            or observation.sequence != index
            or observation.mission_id != reset["mission_id"]
            or observation.map_sha256 != reset["config"]["asset_sha256"]["semantic"]
            or capture.proposal.policy_sha256 != reset["policy_sha256"]
        ):
            raise ValueError("STREAM_CAPTURE_EPISODE_BINDING_MISMATCH")
        temporal = observation.sample.temporal_evidence
        identity = temporal.stream_id, temporal.observed_at_unix_ms
        if identity in identities:
            raise ValueError("STREAM_CAPTURE_REPLAYED_SOURCE")
        identities.add(identity)
        call_id = "model-" + sha256_json(capture.proposal)[:24]
        if call_id in ids:
            raise ValueError("STREAM_CAPTURE_REPLAYED_PROPOSAL")
        ids.append(call_id)
        name = f"stream-proposal-{index:06d}.json"
        raw = decode_json(packed.content, limit=MAXIMUM_CAPTURE_BYTES,
                          node_limit=MAXIMUM_CAPTURE_JSON_NODES)
        source_files[name] = sha256_json(raw)
        write_new(episode / name, raw)
    joined = grounded_control_index(episode / "flight/simulation", set(ids))
    visits, missing = [], []
    for index, (call_id, packed) in enumerate(zip(ids, captures, strict=True)):
        if call_id not in joined:
            missing.append(
                {"sequence": index, "call_id": call_id, "reason": "no-actual-control-acceptance"}
            )
            continue
        capture = packed.unpack()
        command, application = joined[call_id]
        visit = stream_visit(capture, packed, command, application)
        record = {
            "purpose": "grounded-stream-imitation-action",
            "not_a_reward_transition": True,
            "capture": decode_json(packed.content, limit=MAXIMUM_CAPTURE_BYTES,
                                   node_limit=MAXIMUM_CAPTURE_JSON_NODES),
            "capture_sha256": packed.sha256,
            "command": command.model_dump(mode="json"),
            "application": application.model_dump(mode="json"),
            "applied_action": visit.applied_action.model_dump(mode="json"),
            "safety_intervened": visit.safety_intervened,
            "phase": "after-confirmed-native-landing",
            "qualified_for_flight": False,
        }
        name = f"stream-action-{index:06d}.json"
        source_files[name] = sha256_json(record)
        write_new(episode / name, record)
        visits.append(visit)
    write_new(
        episode / "stream-capture-receipt.json",
        {
            "purpose": "grounded-stream-imitation-collection",
            "submitted": len(captures),
            "actually_accepted_decisions": len(visits),
            "unaccepted_proposals": missing,
            "source_content_sha256": source_files,
            "reset_sha256": sha256_json(reset),
            "terminal_sha256": sha256_json(terminal),
            "archive_bytes": sum(len(c.content) for c in captures),
            "not_a_reward_transition": True,
            "qualified_for_flight": False,
        },
    )
    return visits
