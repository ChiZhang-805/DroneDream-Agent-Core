"""Read one content-bound, grounded native episode for offline supervision."""

import re
from dataclasses import dataclass
from pathlib import Path

from dronedream_plugin_sdk.protocol import encode_json

from ..contracts import VehicleAsset
from ..plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    portable_plugin_path,
    read_plugin_file,
)
from .counterfactual_teacher import CounterfactualConfig, CounterfactualTeacher
from .evidence_files import read_evidence_object
from .flight_environment import FlightObservation, FlightStep, PilotAction
from .mission_groups import MissionGroupEvidence, mission_group_evidence
from .native_corrections import GroundedNativeCorrections
from .outcome_verifier import OutcomeEnvelope
from .px4_environment import ASSET_FIELDS, Px4TrainingConfig
from .student_collection import StudentVisit


@dataclass(frozen=True)
class GroundedEpisode:
    """Verified ended transition records and their independently bound teacher context."""
    config: Px4TrainingConfig
    visits: tuple[StudentVisit, ...]
    oracle: GroundedNativeCorrections
    source_files_sha256: dict[str, str]
    mission_group_sha256: str
    mission_split: MissionGroupEvidence

    # 功能：
    #   明确标识本回合为因果奖励步骤，区别于只有动作接纳证据的连续模仿流。
    # 输入：
    #   self：已绑定来源的原生回合。
    # 输出：
    #   kind：因果奖励步骤的固定类别。
    @property
    def native_record_kind(self):
        kind = "reward-step"
        return kind

    # 功能：
    #   发布派生标签前有界复查原始证据和资产，路径遍历或内容变化均阻止发布。
    # 输入：
    #   self：保留原始文件摘要的回合。
    #   path：待复核的回合根目录。
    # 输出：
    #   None：不返回业务数据。
    def verify_sources(self, path: Path) -> None:
        for name, digest in self.source_files_sha256.items():
            portable_plugin_path(name)
            if hash_plugin_file(path / name, limit=4 * 1024 * 1024) != digest:
                raise ValueError("DAGGER_NATIVE_SOURCE_CHANGED:" + name)
        for name, digest in self.config.asset_sha256.items():
            if hash_plugin_file(getattr(self.config, name), limit=256 * 1024 * 1024) != digest:
                raise ValueError("DAGGER_NATIVE_ASSET_CHANGED:" + name)


# 功能：
#   1. 检查落地、原始资产、完整连续步骤及实际动作来源，再构造离线监督回合。
#   2. 拒绝与控制流格式混杂、超预算或歧义证据；所有验证通过仍不等于飞行资格。
# 输入：
#   path：原生回合目录。
#   teacher_config：独立教师的名义动力学配置。
# 输出：
#   episode：带访问记录、独立教师、来源摘要和空间分组的原生回合。
def load_grounded_episode(path: Path, teacher_config: CounterfactualConfig) -> GroundedEpisode:
    if not isinstance(path, Path):
        raise ValueError("DAGGER_NATIVE_PATH_INVALID")
    path = path.absolute()
    check_plain_plugin_path(path)
    terminal_name = "flight/simulation/native-terminal-lifecycle.json"
    terminal, terminal_digest = read_evidence_object(path / terminal_name)
    if not isinstance(terminal, dict) or (
        terminal.get("terminal_state") != "ON_GROUND"
        or terminal.get("landing_confirmed") is not True
        or terminal.get("safe_to_stop_watchdog") is not True
    ):
        raise ValueError("DAGGER_NATIVE_LANDING_NOT_CONFIRMED")
    reset, reset_digest = read_evidence_object(path / "reset.json")
    config = Px4TrainingConfig.model_validate(reset["config"])
    if set(config.asset_sha256) != set(ASSET_FIELDS):
        raise ValueError("DAGGER_NATIVE_ASSET_IDENTITIES_INCOMPLETE")
    for name in ASSET_FIELDS:
        if (hash_plugin_file(getattr(config, name), limit=256 * 1024 * 1024)
                != config.asset_sha256[name]):
            raise ValueError("DAGGER_NATIVE_ASSET_CHANGED:" + name)
    vehicle_row, vehicle_digest = read_evidence_object(config.vehicle, limit=64 * 1024 * 1024)
    semantic, semantic_digest = read_evidence_object(config.semantic, limit=64 * 1024 * 1024)
    if (vehicle_digest != config.asset_sha256["vehicle"]
            or semantic_digest != config.asset_sha256["semantic"]):
        raise ValueError("DAGGER_NATIVE_ASSET_CHANGED_DURING_PARSE")
    vehicle = VehicleAsset.model_validate_json(encode_json(vehicle_row, limit=64 * 1024 * 1024))
    teacher = CounterfactualTeacher(
        semantic.get("runtime_collision_primitives", semantic.get("collision_primitives")),
        OutcomeEnvelope(
            vehicle.body_radius_m, vehicle.body_height_m, config.minimum_enu_m, config.maximum_enu_m
        ),
        teacher_config,
    )
    oracle = GroundedNativeCorrections(
        path.parent,
        teacher,
        map_sha256=config.asset_sha256["semantic"],
        student_policy_sha256=reset["policy_sha256"],
    )
    files = []
    for count, candidate in enumerate(path.iterdir(), start=1):
        if count > 4096:
            raise ValueError("DAGGER_NATIVE_DIRECTORY_LIMIT_EXCEEDED")
        if (candidate.name == "stream-capture-receipt.json"
                or re.fullmatch(r"stream-(?:action|proposal)-\d{6}\.json", candidate.name)):
            raise ValueError("DAGGER_COLLECTION_MODES_CANNOT_BE_MIXED")
        if re.fullmatch(r"transition-\d{6}\.json", candidate.name):
            files.append(candidate)
            if len(files) > 256:
                raise ValueError("DAGGER_NATIVE_TRANSITION_LIMIT_EXCEEDED")
    files.sort()
    visits = []
    hashes = {
        "reset.json": reset_digest,
        terminal_name: terminal_digest,
    }
    for file in files:
        row, hashes[file.name] = read_evidence_object(file, limit=4 * 1024 * 1024)
        observation = FlightObservation.model_validate(row["source_observation"])
        if observation.episode_id != path.name:
            raise ValueError("DAGGER_NATIVE_TRANSITION_EPISODE_MISMATCH")
        if file.name != f"transition-{observation.sequence + 1:06d}.json":
            raise ValueError("DAGGER_NATIVE_TRANSITION_SEQUENCE_MISMATCH")
        visits.append(
            StudentVisit(
                observation,
                PilotAction.model_validate(row["proposal"]["action"]),
                FlightStep.model_validate(row["step"]),
            )
        )
        # 不能仅因步骤各字段可解析就返回训练访问；同时核对真实执行与独立结果的归属。
        visits[-1].step.validate_transition(observation, visits[-1].proposal)
        oracle._context(observation)
    if not visits:
        raise ValueError("DAGGER_NATIVE_COMPLETED_TRANSITIONS_REQUIRED")
    if [visit.observation.sequence for visit in visits] != list(range(len(visits))):
        raise ValueError("DAGGER_NATIVE_TRANSITION_SEQUENCE_GAP")
    # Route/map identity, not user-renamable mission labels or random seeds,
    # defines the holdout group. Two flights of this route stay in one split.
    split = mission_group_evidence(read_plugin_file(config.route, limit=4 * 1024 * 1024),
                                   config.asset_sha256["semantic"],
                                   expected_route_sha256=config.asset_sha256["route"])
    episode = GroundedEpisode(config, tuple(visits), oracle, hashes, split.group_sha256, split)
    episode.verify_sources(path)
    return episode
