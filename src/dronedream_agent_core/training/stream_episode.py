"""Grounded continuous imitation reader and label oracle; no actuation access."""

import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from dronedream_plugin_sdk.protocol import copy_json, encode_json

from ..contracts import RuntimeLocalSafetyCommand, VehicleAsset
from ..control_execution_evidence import ControlApplicationRecord
from ..hashing import sha256_json
from ..local_policy_training import LocalPolicyObservation
from ..plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    portable_plugin_path,
    read_plugin_file,
)
from .counterfactual_teacher import CounterfactualTeacher
from .evidence_files import read_evidence_object
from .flight_environment import FlightObservation
from .grounded_teacher_context import copy_grounded_state, grounded_teacher_context
from .mission_groups import MissionGroupEvidence, mission_group_evidence
from .outcome_verifier import OutcomeEnvelope
from .px4_environment import ASSET_FIELDS, Px4TrainingConfig
from .stream_capture import (
    StreamControlVisit,
    grounded_control_index,
    require_grounded_stream,
    stream_capture_from_payload,
    stream_capture_from_record,
    stream_visit,
)
from .visual_observation import FrozenVisualEncoder


# 功能：
#   用指定冻结编码器重新计算视觉特征，与原始像素身份及特征逐项核对，禁止改写对照样本。
# 输入：
#   capture：已校验的控制流捕获。
#   encoder：冻结视觉编码器，无视觉输入时可为 None。
# 输出：
#   None：不返回业务数据。
def validate_stream_visual(capture, encoder):
    sample = LocalPolicyObservation.model_validate(capture.observation.sample.model_dump())
    media = copy_json(capture.request["multimodal"])
    if encoder is None:
        if media or sample.visual_features or sample.source_visual_sha256:
            raise ValueError("STREAM_ANNOTATION_VISUAL_ENCODER_MISSING")
        return
    supplied = encoder.attach(sample.model_copy(deep=True), media)
    if not isinstance(supplied, LocalPolicyObservation):
        raise ValueError("STREAM_ANNOTATION_VISUAL_RESULT_INVALID")
    supplied = LocalPolicyObservation.model_validate(supplied.model_dump())
    nonvisual = {"visual_features", "source_visual_sha256"}
    if (
        supplied.model_dump(exclude=nonvisual) != sample.model_dump(exclude=nonvisual)
        or supplied.source_visual_sha256 != sample.source_visual_sha256
        or len(supplied.visual_features) != len(sample.visual_features)
        or any(
            abs(a - b) > 1e-6 * max(1.0, abs(a), abs(b))
            for a, b in zip(supplied.visual_features, sample.visual_features, strict=True)
        )
    ):
        raise ValueError("STREAM_ANNOTATION_VISUAL_FEATURES_CHANGED")


# 功能：
#   重建落地后控制流访问，交叉核对回合、策略、原始提案和真实执行回执。
# 输入：
#   row：严格解析的单动作记录。
#   episode：回合目录。
#   reset：已确认的回合初始化记录。
#   index：文件名标识的源观测序号。
# 输出：
#   visit：绑定实际动作且不声称奖励转移的访问。
#   capture：原始不可变捕获解包后的内容。
#   command：原批准命令。
#   application：实际传输回执。
def _validated_visit(row, *, episode, reset, index):
    if (
        row.get("purpose") != "grounded-stream-imitation-action"
        or row.get("not_a_reward_transition") is not True
        or row.get("qualified_for_flight") is not False
        or row.get("phase") != "after-confirmed-native-landing"
    ):
        raise ValueError("STREAM_ANNOTATION_NOT_GROUNDED_IMITATION")
    packed, capture = stream_capture_from_record(row)
    observation = capture.observation
    if (
        observation.episode_id != episode.name
        or observation.sequence != index
        or observation.mission_id != reset["mission_id"]
        or observation.map_sha256 != reset["config"]["asset_sha256"]["semantic"]
        or capture.proposal.policy_sha256 != reset["policy_sha256"]
    ):
        raise ValueError("STREAM_ANNOTATION_EPISODE_MISMATCH")
    command = RuntimeLocalSafetyCommand.model_validate(row["command"])
    application = ControlApplicationRecord.model_validate(row["application"])
    visit = stream_visit(capture, packed, command, application)
    if (
        visit.applied_action.model_dump(mode="json") != row["applied_action"]
        or visit.safety_intervened is not row["safety_intervened"]
    ):
        raise ValueError("STREAM_ANNOTATION_ACTUAL_ACTION_MISMATCH")
    return visit, capture, command, application


class GroundedStreamCorrections:
    """Offline correction oracle whose cached contexts remain bound to current source files."""

    # 功能：
    #   固定本回合目录及来源摘要清单，建立只供离线标注的单上下文缓存。
    # 输入：
    #   self：连续流教师纠正器。
    #   episode：本回合目录。
    #   teacher：独立几何与名义动力学教师。
    #   source_files_sha256：回合内规范相对路径到原始字节摘要的映射。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, episode, teacher, source_files_sha256):
        if not isinstance(episode, Path) or type(source_files_sha256) is not dict:
            raise ValueError("STREAM_ANNOTATION_SOURCE_INVENTORY_INVALID")
        if not 1 <= len(source_files_sha256) <= 1024:
            raise ValueError("STREAM_ANNOTATION_SOURCE_INVENTORY_INVALID")
        for name, digest in source_files_sha256.items():
            portable_plugin_path(name)
            if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("STREAM_ANNOTATION_SOURCE_INVENTORY_INVALID")
        self.episode, self.teacher = episode.absolute(), teacher
        check_plain_plugin_path(self.episode)
        self.sources = dict(source_files_sha256)
        self.receipts = []
        self._key = self._state = None

    # 功能：
    #   每次重查落地、初始化及动作文件，按当前教师配置缓存独立上下文并返回副本。
    # 输入：
    #   self：来源约束和上下文缓存。
    #   observation：原始学生观测。
    # 输出：
    #   state：仅用于监督标签的独立教师状态。
    def _context(self, observation):
        if not isinstance(observation, FlightObservation):
            raise ValueError("STREAM_ANNOTATION_OBSERVATION_INVALID")
        observation = FlightObservation.model_validate(observation.model_dump())
        reset, terminal = require_grounded_stream(self.episode)
        name = f"stream-action-{observation.sequence:06d}.json"
        row, content_digest = read_evidence_object(self.episode / name, limit=8 * 1024 * 1024)
        if content_digest != self.sources.get(name):
            raise ValueError("STREAM_ANNOTATION_SOURCE_CHANGED")
        for filename in ("reset.json", "flight/simulation/native-terminal-lifecycle.json"):
            if (
                _file_hash(self.episode / filename)
                != self.sources[filename]
            ):
                raise ValueError("STREAM_ANNOTATION_COLLECTION_CHANGED")
        key = (sha256_json(observation), self.sources[name], sha256_json(self.teacher.config),
               self.teacher.geometry_sha256)
        if key == self._key:
            state = copy_grounded_state(self._state)
            return state
        visit, capture, _, application = _validated_visit(
            row, episode=self.episode, reset=reset, index=observation.sequence
        )
        if visit.observation != observation:
            raise ValueError("STREAM_ANNOTATION_OBSERVATION_CHANGED")
        state, context = grounded_teacher_context(
            observation,
            capture.request["snapshot"],
            capture.initial_witness,
            application,
            self.teacher,
            binding={
                "capture_sha256": visit.capture_sha256,
                "terminal_receipt_sha256": sha256_json(terminal),
                "reset_sha256": sha256_json(reset),
            },
        )
        self.receipts.append({"context": context, "context_sha256": state.context_sha256})
        self._key, self._state = key, state
        state = copy_grounded_state(state)
        return state

    # 功能：
    #   基于原始观测及独立上下文生成纠正，并保留可追溯教师回执。
    # 输入：
    #   self：离线连续流纠正器。
    #   observation：原始学生观测。
    # 输出：
    #   correction：与原观测身份绑定的教师动作标签。
    def correction(self, observation):
        correction, receipt = self.teacher.correction(observation, self._context(observation))
        self.receipts.append({"receipt": receipt, "receipt_sha256": sha256_json(receipt)})
        return correction

    # 功能：
    #   单独评价学生提案的风险，不以教师替代动作的评价冒充原提案的评价。
    # 输入：
    #   self：离线连续流纠正器。
    #   observation：原始学生观测。
    #   action：待评价的学生提案。
    # 输出：
    #   risk：名义反事实风险标签。
    def risk(self, observation, action):
        risk, receipt = self.teacher.risk(observation, self._context(observation), action)
        if receipt is not None:
            self.receipts.append({"receipt": receipt, "receipt_sha256": sha256_json(receipt)})
        return risk


# 功能：
#   流式计算有界普通文件的原始字节摘要，并检查读取期间的文件身份和大小变化。
# 输入：
#   path：证据或资产文件路径。
# 输出：
#   digest：至多 256 MiB 文件的 SHA-256 摘要。
def _file_hash(path):
    digest = hash_plugin_file(path, limit=256 * 1024 * 1024)
    return digest


@dataclass(frozen=True)
class GroundedStreamEpisode:
    """Source-bound streaming imitation visits; deliberately no reward-transition API."""
    config: Px4TrainingConfig
    visits: tuple[StreamControlVisit, ...]
    oracle: GroundedStreamCorrections
    source_files_sha256: dict[str, str]
    mission_split: MissionGroupEvidence

    # 功能：
    #   提供按语义地图和空间路线计算的分组身份，不使用可随意改名的任务标题。
    # 输入：
    #   self：已绑定来源的控制流回合。
    # 输出：
    #   digest：任务空间分组摘要。
    @property
    def mission_group_sha256(self):
        digest = self.mission_split.group_sha256
        return digest

    # 功能：
    #   声明本回合是连续流动作记录，不能被当作单动作因果奖励转移。
    # 输入：
    #   self：当前控制流回合。
    # 输出：
    #   kind：固定的控制流动作记录类别。
    @property
    def native_record_kind(self):
        kind = "stream-action"
        return kind

    # 功能：
    #   在发布派生标签前重新核对全部源文件与资产，拒绝篡改及不安全相对路径。
    # 输入：
    #   self：包含预期证据和资产身份的回合。
    #   path：需重新核对的回合目录。
    # 输出：
    #   None：不返回业务数据。
    def verify_sources(self, path):
        for name, digest in self.source_files_sha256.items():
            portable_plugin_path(name)
            if _file_hash(path / name) != digest:
                raise ValueError("STREAM_ANNOTATION_SOURCE_CHANGED:" + name)
        for name, digest in self.config.asset_sha256.items():
            if _file_hash(getattr(self.config, name)) != digest:
                raise ValueError("STREAM_ANNOTATION_ASSET_CHANGED:" + name)


# 功能：
#   持有并验证回合绑定的离线视觉编码器，所有成功和异常退出路径都关闭资源。
# 输入：
#   config：回合配置及可选视觉包路径。
#   reset：采集时记录的编码器身份和输入契约。
# 输出：
#   visual：上下文内可用的冻结编码器或 None。
@contextmanager
def _bound_visual_encoder(config, reset):
    visual = FrozenVisualEncoder(config.visual_package) if config.visual_package else None
    try:
        if (reset.get("visual_encoder_sha256") != (visual.sha256 if visual else None)
                or reset.get("visual_input_contract")
                != (visual.input_contract if visual else None)):
            raise ValueError("STREAM_ANNOTATION_VISUAL_BINDING_CHANGED")
        yield visual
    finally:
        if visual is not None:
            visual.close()


# 功能：
#   1. 验证落地后声明的完整控制流、资产、实际账本及提案接纳关系。
#   2. 重新核对视觉和分组来源，形成可供离线标注的回合，不授予飞行资格。
# 输入：
#   path：待加载的回合目录。
#   teacher_config：独立教师使用的名义动力学配置。
# 输出：
#   episode：携带原始访问、教师纠正器、文件摘要及空间分组的回合。
def load_grounded_stream_episode(path: Path, teacher_config) -> GroundedStreamEpisode:
    if not isinstance(path, Path):
        raise ValueError("STREAM_ANNOTATION_PATH_INVALID")
    path = path.absolute()
    check_plain_plugin_path(path)
    reset, terminal = require_grounded_stream(path)
    config = Px4TrainingConfig.model_validate(reset["config"])
    if set(config.asset_sha256) != set(ASSET_FIELDS):
        raise ValueError("STREAM_ANNOTATION_ASSET_IDENTITIES_INCOMPLETE")
    for name in ASSET_FIELDS:
        if _file_hash(getattr(config, name)) != config.asset_sha256[name]:
            raise ValueError("STREAM_ANNOTATION_ASSET_CHANGED:" + name)
    vehicle_row, vehicle_hash = read_evidence_object(config.vehicle, limit=64 * 1024 * 1024)
    semantic, semantic_hash = read_evidence_object(config.semantic, limit=64 * 1024 * 1024)
    if (vehicle_hash != config.asset_sha256["vehicle"]
            or semantic_hash != config.asset_sha256["semantic"]):
        raise ValueError("STREAM_ANNOTATION_ASSET_CHANGED_DURING_PARSE")
    vehicle = VehicleAsset.model_validate_json(encode_json(vehicle_row, limit=64 * 1024 * 1024))
    teacher = CounterfactualTeacher(
        semantic.get("runtime_collision_primitives", semantic.get("collision_primitives")),
        OutcomeEnvelope(
            vehicle.body_radius_m, vehicle.body_height_m, config.minimum_enu_m, config.maximum_enu_m
        ),
        teacher_config,
    )
    receipt, receipt_hash = read_evidence_object(path / "stream-capture-receipt.json")
    if (
        receipt.get("purpose") != "grounded-stream-imitation-collection"
        or receipt.get("not_a_reward_transition") is not True
        or receipt.get("qualified_for_flight") is not False
        or receipt.get("reset_sha256") != sha256_json(reset)
        or receipt.get("terminal_sha256") != sha256_json(terminal)
        or type(receipt.get("submitted")) is not int
        or not 1 <= receipt["submitted"] <= 256
        or type(receipt.get("actually_accepted_decisions")) is not int
    ):
        raise ValueError("STREAM_ANNOTATION_COLLECTION_RECEIPT_INVALID")
    declared = receipt.get("source_content_sha256")
    if (
        not isinstance(declared, dict)
        or not declared
        or len(declared) > 512
        or any(
            not re.fullmatch(r"stream-(?:action|proposal)-\d{6}\.json", name) for name in declared
        )
    ):
        raise ValueError("STREAM_ANNOTATION_SOURCE_INVENTORY_INVALID")
    actual = set()
    for count, candidate in enumerate(path.iterdir(), start=1):
        if count > 4096:
            raise ValueError("STREAM_ANNOTATION_DIRECTORY_LIMIT_EXCEEDED")
        if re.fullmatch(r"stream-(?:action|proposal)-\d{6}\.json", candidate.name):
            actual.add(candidate.name)
    if actual != set(declared):
        raise ValueError("STREAM_ANNOTATION_SOURCE_INVENTORY_CHANGED")
    names = [
        "reset.json",
        "stream-capture-receipt.json",
        "flight/simulation/native-terminal-lifecycle.json",
        "flight/simulation/depth-local-safety-history.jsonl",
        "flight/simulation/runtime-state/control-applications.jsonl",
        "flight/simulation/runtime-state/control-application-writer.json",
        "flight/simulation/runtime-evidence-writer-summary.json",
    ]
    sources = {name: _file_hash(path / name) for name in names}
    # 解析结果和保存的摘要必须是同一内容，不能先解析旧文件再给替换后的文件记摘要。
    if sources["stream-capture-receipt.json"] != receipt_hash:
        raise ValueError("STREAM_ANNOTATION_COLLECTION_CHANGED")
    for name, expected in (("reset.json", reset),
                           ("flight/simulation/native-terminal-lifecycle.json", terminal)):
        parsed, digest = read_evidence_object(path / name)
        if parsed != expected or digest != sources[name]:
            raise ValueError("STREAM_ANNOTATION_COLLECTION_CHANGED")
    visits, joins, proposals = [], {}, {}
    with _bound_visual_encoder(config, reset) as visual:
        for name in sorted(declared):
            row, content_hash = read_evidence_object(path / name, limit=8 * 1024 * 1024)
            if sha256_json(row) != declared[name]:
                raise ValueError("STREAM_ANNOTATION_SOURCE_CONTENT_CHANGED")
            sources[name] = content_hash
            if not name.startswith("stream-action-"):
                capture = stream_capture_from_payload(row)
                index = int(name[-11:-5])
                if (
                    capture.observation.episode_id != path.name
                    or capture.observation.sequence != index
                    or not 0 <= index < receipt["submitted"]
                    or capture.proposal.policy_sha256 != reset["policy_sha256"]
                ):
                    raise ValueError("STREAM_ANNOTATION_PROPOSAL_IDENTITY_INVALID")
                call_id = "model-" + sha256_json(capture.proposal)[:24]
                if call_id in proposals:
                    raise ValueError("STREAM_ANNOTATION_PROPOSAL_DUPLICATED")
                proposals[call_id] = index
                continue
            index = int(name[-11:-5])
            visit, capture, command, application = _validated_visit(
                row, episode=path, reset=reset, index=index
            )
            proposal_name = f"stream-proposal-{index:06d}.json"
            if (proposal_name not in declared
                    or sha256_json(row["capture"]) != declared[proposal_name]):
                raise ValueError("STREAM_ANNOTATION_PROPOSAL_MISMATCH")
            visits.append(visit)
            validate_stream_visual(capture, visual)
            joins[command.model_call_id] = command, application
    if not visits or len(visits) != receipt.get("actually_accepted_decisions"):
        raise ValueError("STREAM_ANNOTATION_ACCEPTED_DECISIONS_REQUIRED")
    # Rejoin the original durable runtime ledger, not merely locally claimed
    # applications inside a newly edited annotation file.
    actual_joins = grounded_control_index(path / "flight/simulation", set(proposals))
    if actual_joins != joins:
        raise ValueError("STREAM_ANNOTATION_ACTUAL_LEDGER_CHANGED")
    missing = [
        {"sequence": index, "call_id": call_id, "reason": "no-actual-control-acceptance"}
        for call_id, index in proposals.items()
        if call_id not in actual_joins
    ]
    if receipt.get("unaccepted_proposals") != missing:
        raise ValueError("STREAM_ANNOTATION_UNACCEPTED_INVENTORY_CHANGED")
    if {f"stream-proposal-{i:06d}.json" for i in range(receipt["submitted"])} != {
        name for name in declared if name.startswith("stream-proposal-")
    }:
        raise ValueError("STREAM_ANNOTATION_PROPOSAL_SEQUENCE_INCOMPLETE")
    split = mission_group_evidence(
        read_plugin_file(config.route, limit=4 * 1024 * 1024),
        config.asset_sha256["semantic"],
        expected_route_sha256=config.asset_sha256["route"],
    )
    episode = GroundedStreamEpisode(
        config, tuple(visits), GroundedStreamCorrections(path, teacher, sources), sources, split
    )
    episode.verify_sources(path)
    return episode


# 功能：
#   按回合自身记录选择连续流或因果步骤读取器，两种必要数据类型不能在同回合混用。
# 输入：
#   path：待加载回合目录。
#   teacher_config：离线教师配置。
# 输出：
#   episode：显式类别的离线模仿回合。
def load_grounded_imitation_episode(path: Path, teacher_config):
    check_plain_plugin_path(path)
    streaming = (path / "stream-capture-receipt.json").is_file()
    if streaming:
        if any(path.glob("transition-[0-9]*.json")):
            raise ValueError("DAGGER_COLLECTION_MODES_CANNOT_BE_MIXED")
        episode = load_grounded_stream_episode(path, teacher_config)
        return episode
    from .native_episode import load_grounded_episode

    episode = load_grounded_episode(path, teacher_config)
    return episode
