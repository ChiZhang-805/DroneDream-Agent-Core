"""Content-bound, risk-only data admission; no flight or actor optimization.

A probe is a hypothetical control, not an imitation target. All controls from
one observed state stay together; all flights of one route/map stay together.
Source poses in teacher receipts are never passed to the neural input arrays.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..contracts import NormalizedPilotControl
from ..hashing import sha256_json
from ..local_policy_training import LocalPolicyTrainingSample
from ..pilot_control_mapping import action_risk_features, physical_pilot_request
from .action_risk_dataset import bounded_action_probes
from .counterfactual_teacher import CounterfactualConfig
from .dagger import ProposedActionRisk
from .evidence_files import decode_evidence_rows, read_evidence_dataset, read_evidence_object
from .flight_environment import FlightObservation, PilotAction
from .mission_groups import MissionGroupEvidence, MissionGroupManifest

FILES = {
    name: name + ".jsonl"
    for name in (
        "action-risk",
        "action-records",
        "counterfactual-receipts",
        "source-observations",
    )
} | {"stream-groups": "stream-groups.json"}
AXES = ("forward_axis", "right_axis", "up_axis", "yaw_axis")


@dataclass(frozen=True)
class ActionRiskDataset:
    """Risk-only samples and their source evidence; never a behavior-cloning corpus."""
    samples: tuple[LocalPolicyTrainingSample, ...]
    records: tuple[dict, ...]
    observations: tuple[FlightObservation, ...]
    groups: frozenset[str]
    receipt_sha256: str
    teacher_config_sha256: str
    receipt: dict


# 功能：
#   使用与行为数据集一致的严格完整行解析，不在摘要核对之后重新读取文件。
# 输入：
#   content：已固定且验证摘要的 JSONL 字节。
# 输出：
#   rows：符合大小、行数和对象约束的记录列表。
def _rows(content):
    rows = decode_evidence_rows(content)
    return rows


# 功能：
#   检查来源引用必须是小写 SHA-256 字符串，拒绝空值或非摘要身份。
# 输入：
#   value：待核对摘要。
# 输出：
#   value：通过检查的原摘要。
def _hash(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("ACTION_RISK_IDENTITY_INVALID")
    return value


# 功能：
#   将每回合选择的观测序号绑定地图、路线及记录类别，禁止用改名或混合类别逃避来源检查。
# 输入：
#   receipt：数据集完成回执。
# 输出：
#   sources：按唯一回合标识索引的来源记录。
def _sources(receipt):
    sources = {}
    for source in receipt["source_receipts"]:
        episode = source["episode_id"]
        if not episode or episode in sources:
            raise ValueError("ACTION_RISK_SOURCE_EPISODE_DUPLICATE")
        assets = source["asset_sha256"]
        split = MissionGroupEvidence.model_validate(source.get("mission_split"))
        if (split.group_sha256 != source["mission_group_sha256"]
                or split.route_sha256 != _hash(assets["route"])
                or split.semantic_sha256 != _hash(assets["semantic"])):
            raise ValueError("ACTION_RISK_ROUTE_GROUP_MISMATCH")
        files = source["source_files_sha256"]
        for name in ("reset.json", "flight/simulation/native-terminal-lifecycle.json"):
            _hash(files[name])
        sequences = source["selected_sequences"]
        if (
            not sequences
            or any(type(seq) is not int or seq < 0 for seq in sequences)
            or (sorted(set(sequences)) != sequences)
        ):
            raise ValueError("ACTION_RISK_SOURCE_SEQUENCES_INVALID")
        kind = source.get("native_record_kind", "reward-step")
        if kind not in {"reward-step", "stream-action"}:
            raise ValueError("ACTION_RISK_NATIVE_RECORD_KIND_INVALID")
        if kind == "stream-action":
            _hash(files.get("stream-capture-receipt.json"))
            if any(name.startswith("transition-") for name in files):
                raise ValueError("ACTION_RISK_NATIVE_RECORD_KINDS_MIXED")
        elif "stream-capture-receipt.json" in files:
            raise ValueError("ACTION_RISK_NATIVE_RECORD_KINDS_MIXED")
        for seq in sequences:
            if kind == "stream-action":
                _hash(files.get(f"stream-action-{seq:06d}.json"))
                _hash(files.get(f"stream-proposal-{seq:06d}.json"))
            else:
                _hash(files.get(f"transition-{seq + 1:06d}.json"))
        sources[episode] = source
    if not sources:
        raise ValueError("ACTION_RISK_SOURCES_REQUIRED")
    return sources


# 功能：
#   重新计算独立上下文和反事实预测回执的摘要，拒绝内容错绑或同类重复身份。
# 输入：
#   rows：严格解析的教师回执记录。
# 输出：
#   contexts：上下文摘要到原上下文的映射。
#   predictions：预测摘要到反事实计算结果的映射。
def _witnesses(rows):
    contexts, predictions = {}, {}
    for row in rows:
        if "context" in row:
            context, digest = row["context"], row["context_sha256"]
            if sha256_json(context) != digest or digest in contexts:
                raise ValueError("ACTION_RISK_CONTEXT_BINDING_MISMATCH")
            contexts[digest] = context
        else:
            prediction, digest = row["receipt"], row["receipt_sha256"]
            if sha256_json(prediction) != digest or digest in predictions:
                raise ValueError("ACTION_RISK_PREDICTION_BINDING_MISMATCH")
            predictions[digest] = prediction
    return contexts, predictions


# 功能：
#   1. 对照同次读取的原始字节摘要接纳风险专用数据，并限制整个数据集与每行大小。
#   2. 核对观测、动作探针、物理尺度、独立预测和路线分组，禁止将风险探针当成行为标签。
# 输入：
#   root：风险数据集目录。
# 输出：
#   dataset：来源和标签对应关系验证后的风险数据集。
def load_action_risk_dataset(root: Path) -> ActionRiskDataset:
    receipt, receipt_digest = read_evidence_object(root / "dataset-receipt.jsonl")
    if not isinstance(receipt, dict) or (
        receipt.get("purpose") != "grounded-native-action-risk-supervision"
        or receipt.get("behavior_cloning_dataset") is not False
        or receipt.get("qualified_for_flight") is not False
        or receipt.get("counterfactual_source") != "nominal-swept-geometry-not-physical-replay"
    ):
        raise ValueError("ACTION_RISK_OFFLINE_RISK_ONLY_DATA_REQUIRED")
    config = CounterfactualConfig.model_validate(receipt["teacher_config"])
    contents = read_evidence_dataset(root, FILES, receipt.get("file_sha256"),
                                     error_prefix="ACTION_RISK_DATASET_CONTENT_CHANGED")
    samples = tuple(
        LocalPolicyTrainingSample.model_validate(row) for row in _rows(contents["action-risk"])
    )
    records = tuple(_rows(contents["action-records"]))
    observations = tuple(
        FlightObservation.model_validate(row) for row in _rows(contents["source-observations"])
    )
    sources = _sources(receipt)
    group_rows = _rows(contents["stream-groups"])
    if len(group_rows) != 1:
        raise ValueError("ACTION_RISK_GROUP_MANIFEST_AMBIGUOUS")
    groups = MissionGroupManifest.model_validate(group_rows[0]).groups
    contexts, predictions = _witnesses(_rows(contents["counterfactual-receipts"]))
    actions = bounded_action_probes()
    probes = {sha256_json(action) for action in actions}
    if (
        receipt.get("probe_actions_sha256") != sha256_json(actions)
        or receipt.get("probe_count_per_observation") != len(probes)
        or not samples
        or len(samples) != len(records)
        or len(samples) != receipt.get("sample_count")
        or len(samples) != len(observations) * len(probes)
    ):
        raise ValueError("ACTION_RISK_DATASET_COUNT_OR_PROBES_INVALID")
    observed, selected, streams = {}, set(), {}
    for observation in observations:
        digest = sha256_json(observation)
        identity = observation.episode_id, observation.sequence
        source = sources.get(observation.episode_id)
        if (
            digest in observed
            or identity in selected
            or source is None
            or (
                observation.sequence not in source["selected_sequences"]
                or observation.map_sha256 != source["asset_sha256"]["semantic"]
            )
        ):
            raise ValueError("ACTION_RISK_OBSERVATION_SOURCE_MISMATCH")
        stream = observation.sample.temporal_evidence.stream_id
        group = source["mission_group_sha256"]
        if streams.setdefault(stream, group) != group:
            raise ValueError("ACTION_RISK_STREAM_GROUP_CONFLICT")
        observed[digest] = observation
        selected.add(identity)
    if (
        selected
        != {(ep, seq) for ep, source in sources.items() for seq in source["selected_sequences"]}
        or streams != groups
    ):
        raise ValueError("ACTION_RISK_SOURCE_COVERAGE_MISMATCH")
    used, used_predictions, used_contexts = set(), set(), set()
    # A source appears once per hypothetical action. Match physical limits and
    # all teacher evidence; identical array shapes alone do not bind a label.
    for label, record in zip(samples, records, strict=True):
        digest = record["observation_sha256"]
        observation = observed.get(digest)
        action = PilotAction.model_validate(record["proposed_action"])
        risk = ProposedActionRisk.model_validate(record["assessment"])
        key = digest, sha256_json(action)
        if (
            observation is None
            or key in used
            or key[1] not in probes
            or record.get("counterfactual_action_executed") is not False
            or record.get("behavior_cloning_label") is not False
            or risk.source != "swept-geometry"
            or risk.observation_sha256 != digest
            or risk.proposed_action_sha256 != key[1]
            or (record["episode_id"], record["sequence"])
            != (observation.episode_id, observation.sequence)
        ):
            raise ValueError("ACTION_RISK_PROPOSAL_SOURCE_MISMATCH")
        used.add(key)
        limits = observation.sample.pilot_control_limits
        if limits is None:
            raise ValueError("ACTION_RISK_PHYSICAL_LIMITS_REQUIRED")
        control = NormalizedPilotControl(**dict(zip(AXES, action.axes, strict=True)))
        expected = LocalPolicyTrainingSample.model_validate(
            {
                **observation.sample.model_dump(),
                "target_action_index": 11,
                "target_pilot_control": action.axes,
                "risk_target": risk.risk,
                "risk_proposed_control": list(
                    action_risk_features(control, limits, harness_scale=1.0)
                ),
            }
        )
        if label != expected or record["label_sha256"] != sha256_json(expected):
            raise ValueError("ACTION_RISK_LABEL_OR_PHYSICAL_CONTROL_MISMATCH")
        prediction = predictions.get(risk.verifier_receipt_sha256)
        if prediction is None:
            raise ValueError("ACTION_RISK_PREDICTION_MISSING")
        context = contexts.get(prediction["context_sha256"])
        source = sources[observation.episode_id]
        if (
            prediction.get("purpose") != "offline-nominal-swept-geometry"
            or prediction.get("qualification_granted") is not False
            or prediction.get("not_a_calibrated_collision_probability") is not True
            or prediction["risk_target"] != risk.risk
            or prediction["action"] != action.model_dump()
            or prediction["geometry_sha256"] != source["teacher_geometry_sha256"]
            or sha256_json(prediction["vehicle_envelope"]) != source["vehicle_envelope_sha256"]
            or prediction["config"] != config.model_dump()
            or prediction["physical_request"]
            != list(physical_pilot_request(control, limits, harness_scale=1.0))
            or context is None
            or context["source_observation_sha256"] != digest
            or context["source_snapshot_sha256"] != observation.sample.source_snapshot_sha256
            or context["source_ms"] != observation.sample.temporal_evidence.observed_at_unix_ms
        ):
            raise ValueError("ACTION_RISK_PREDICTION_CONTEXT_MISMATCH")
        used_predictions.add(risk.verifier_receipt_sha256)
        used_contexts.add(prediction["context_sha256"])
    if used_predictions != set(predictions) or used_contexts != set(contexts):
        raise ValueError("ACTION_RISK_ORPHAN_COUNTERFACTUAL_RECEIPT")
    counts = dict(Counter("unsafe" if row.risk_target >= 0.5 else "safe" for row in samples))
    if counts != receipt.get("class_counts"):
        raise ValueError("ACTION_RISK_CLASS_COUNTS_MISMATCH")
    dataset = ActionRiskDataset(
        samples,
        records,
        observations,
        frozenset(groups.values()),
        receipt_digest,
        sha256_json(config),
        receipt,
    )
    return dataset


# 功能：
#   检查训练与留出数据按整条地图路线隔离，没有重复来源且使用相同教师动力学配置。
# 输入：
#   training：训练侧风险数据集列表。
#   validation：独立留出侧风险数据集列表。
# 输出：
#   None：不返回业务数据。
def validate_action_risk_splits(training, validation):
    if not training or not validation:
        raise ValueError("ACTION_RISK_TWO_INDEPENDENT_SPLITS_REQUIRED")
    seen_receipts, seen_observations, split_groups = set(), set(), []
    configs = set()
    for datasets in (training, validation):
        groups = set()
        for dataset in datasets:
            if dataset.receipt_sha256 in seen_receipts:
                raise ValueError("ACTION_RISK_DATASET_REPEATED")
            seen_receipts.add(dataset.receipt_sha256)
            for observation in dataset.observations:
                identity = observation.episode_id, observation.sequence
                # Two copies of a source cannot become more independent evidence.
                if identity in seen_observations:
                    raise ValueError("ACTION_RISK_SOURCE_OBSERVATION_REPEATED")
                seen_observations.add(identity)
            groups.update(dataset.groups)
            configs.add(dataset.teacher_config_sha256)
        split_groups.append(groups)
    if len(configs) != 1:
        raise ValueError("ACTION_RISK_PHYSICS_SCOPE_MISMATCH")
    if split_groups[0] & split_groups[1]:
        raise ValueError("ACTION_RISK_ROUTE_MAP_HOLDOUT_OVERLAP")
