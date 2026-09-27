"""Re-run nominal geometry for a decision-time latency envelope; preserve capture evidence."""

from collections import Counter
from pathlib import Path

from ..contracts import DynamicObstacleObservation, QuaternionWxyz, Vector3
from ..hashing import sha256_json
from ..plugin_files import check_plain_plugin_path
from .action_risk_artifacts import FILES, load_action_risk_dataset
from .counterfactual_teacher import CounterfactualConfig, CounterfactualState, CounterfactualTeacher
from .dagger_artifacts import write_rows
from .evidence_files import decode_evidence_rows, read_evidence_dataset
from .flight_environment import PilotAction
from .outcome_verifier import OutcomeEnvelope
from .risk_clearance_contract import OBSERVED_CLEARANCE_LABELS


# 功能：逐动作重新积分、查图和标注，使用决策前确定的延迟包络，绝不把旧轨迹只改个版本串。
# 输入：source：完整旧风险集；destination：新目录。
#   primitives：必须匹配原教师摘要的地图；progress：可选进度回调。
# 输出：receipt：新的独立离线预测数据回执；传感器、原执行证据及训练/验证分组不变，不授予飞行资格。
def reannotate_decision_latency(source: Path, destination: Path, primitives, *, progress=None):
    source, destination = source.absolute(), destination.absolute()
    check_plain_plugin_path(source)
    check_plain_plugin_path(destination)
    if source == destination or source in destination.parents or destination.exists():
        raise ValueError("ACTION_RISK_REANNOTATION_DESTINATION_MUST_BE_NEW")
    dataset = load_action_risk_dataset(source)
    config = CounterfactualConfig.model_validate(dataset.receipt["teacher_config"])
    if (
        config.decision_latency_policy != "measured-actuation-v1"
        or config.risk_label_semantics != OBSERVED_CLEARANCE_LABELS
    ):
        raise ValueError("ACTION_RISK_DECISION_LATENCY_SOURCE_INVALID")
    geometry_sha = sha256_json(primitives)
    if any(
        row["teacher_geometry_sha256"] != geometry_sha for row in dataset.receipt["source_receipts"]
    ):
        raise ValueError("ACTION_RISK_DECISION_LATENCY_GEOMETRY_MISMATCH")
    contents = read_evidence_dataset(
        source,
        FILES,
        dataset.receipt["file_sha256"],
        error_prefix="ACTION_RISK_DATASET_CONTENT_CHANGED",
    )
    rows = {name: decode_evidence_rows(content) for name, content in contents.items()}
    old_predictions = {
        row["receipt_sha256"]: row["receipt"]
        for row in rows["counterfactual-receipts"]
        if "receipt" in row
    }
    predictions = [row for row in rows["counterfactual-receipts"] if "context" in row]
    observations = {sha256_json(obs): obs for obs in dataset.observations}
    config.decision_latency_policy = "bounded-source-age-v1"
    teachers, changed = {}, 0
    for index, (label, record) in enumerate(
        zip(rows["action-risk"], rows["action-records"], strict=True)
    ):
        previous = old_predictions[record["assessment"]["verifier_receipt_sha256"]]
        envelope = previous["vehicle_envelope"]
        key = sha256_json(envelope)
        if key not in teachers:
            teachers[key] = CounterfactualTeacher(primitives, OutcomeEnvelope(**envelope), config)
        values = previous["state"]
        state = CounterfactualState(
            position=Vector3.model_validate(values["position"]),
            velocity=Vector3.model_validate(values["velocity"]),
            orientation=QuaternionWxyz.model_validate(values["orientation"]),
            goal=Vector3.model_validate(values["goal"]),
            dynamic_obstacles=tuple(
                DynamicObstacleObservation.model_validate(o) for o in values["dynamic_obstacles"]
            ),
            context_sha256=previous["context_sha256"],
            additional_position_uncertainty_m=values["additional_position_uncertainty_m"],
            observed_action_latency_seconds=values["observed_action_latency_seconds"],
        )
        observation = observations[record["observation_sha256"]]
        assessment, prediction = teachers[key].risk(
            observation, state, PilotAction.model_validate(record["proposed_action"])
        )
        if assessment is None or prediction is None:
            raise ValueError("ACTION_RISK_DECISION_LATENCY_NONCONTINUOUS_PROBE")
        changed += label["risk_target"] != assessment.risk
        label["risk_target"] = assessment.risk
        record["assessment"] = assessment.model_dump(mode="json")
        record["label_sha256"] = sha256_json(label)
        predictions.append(
            dict(receipt=prediction, receipt_sha256=assessment.verifier_receipt_sha256)
        )
        if progress is not None and (index % 161 == 160 or index + 1 == len(rows["action-risk"])):
            progress(index + 1, len(rows["action-risk"]))
    rows["counterfactual-receipts"] = predictions
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {name: write_rows(destination / FILES[name], value) for name, value in rows.items()}
    receipt = {
        **dataset.receipt,
        "teacher_config": config.model_dump(),
        "file_sha256": hashes,
        "class_counts": dict(
            Counter(
                "unsafe" if row["risk_target"] >= 0.5 else "safe" for row in rows["action-risk"]
            )
        ),
        "reannotation": dict(
            kind="decision-latency-envelope-reprediction-v1",
            source_dataset_receipt_sha256=dataset.receipt_sha256,
            source_file_sha256=dataset.receipt["file_sha256"],
            previous_teacher_config=dataset.receipt["teacher_config"],
            previous_reannotation=dataset.receipt.get("reannotation"),
            changed_label_count=changed,
            new_physical_observation_count=0,
            changed_sensor_features=False,
            changed_physical_prediction=True,
            changed_route_groups=False,
            behavior_cloning_dataset=False,
        ),
    }
    write_rows(destination / "dataset-receipt.jsonl", [receipt])
    admitted = load_action_risk_dataset(destination)
    if admitted.observations != dataset.observations or admitted.groups != dataset.groups:
        raise ValueError("ACTION_RISK_REANNOTATION_SOURCE_DRIFT")
    return receipt
